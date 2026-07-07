import csv
import os
import shutil
import site
import sys

# When running with a local ultralytics/ directory present, Python would normally
# pick up the local folder instead of the conda-installed package. We fix this by
# manipulating sys.path explicitly — since insert(0, ...) is a stack operation,
# entries are added in reverse priority order so that conda site-packages lands
# at index 0 and takes precedence over the local directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # index 1 — local imports
sys.path.insert(0, site.getsitepackages()[0])  # index 0 — conda site-packages (priority)

from ultralytics.utils import LOGGER
from ultralytics.utils import SETTINGS, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers
from ultralytics.utils.callbacks.wb import _plot_curve, _log_plots

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["wandb"] is True  # verify integration is enabled
    import wandb as wb

    assert hasattr(wb, "__version__")  # verify package is not directory

except (ImportError, AssertionError):
    wb = None

SETTINGS["wandb"] = False
VERBOSE = 0


def eval_last(trainer):
    if trainer.last.exists():
        LOGGER.info(f"\nValidating {trainer.last}...")
        metrics = trainer.validator(model=trainer.last)

        for csv_path in [
            trainer.save_dir.parent / "results.csv",
            # trainer.save_dir.parent.parent / "results/results.csv",
            # trainer.save_dir.parent.parent / "results.csv"
        ]:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            row = {"name": trainer.args.name, **{k: round(v, 3) for k, v in metrics.items()}}
            write_header = not csv_path.exists()
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            LOGGER.info(f"Results saved to {csv_path}")


def move_last_ckpt(trainer):
    src = trainer.last
    dst = trainer.save_dir / "last.pt"
    shutil.move(str(src), str(dst))
    trainer.last = dst


def on_pretrain_routine_start(trainer):
    """Initialize and start wandb project if module is present."""
    if not wb.run:
        import tempfile
        from datetime import datetime
        from pathlib import Path

        wandb_dir = Path(tempfile.gettempdir())

        name = str(trainer.args.name).replace("/", "-").replace(" ", "_")
        latest_run = wandb_dir / "wandb" / "latest-run"
        resuming = trainer.args.resume and latest_run.exists()
        wb.init(
            project=str(trainer.args.project).replace("/", "-") if trainer.args.project else "Ultralytics",
            name=name,
            config=vars(trainer.args),
            id=latest_run.resolve().name.split("-", 2)[2]
            if resuming
            else f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            resume="allow" if resuming else None,
            dir=str(wandb_dir),
        )


def on_fit_epoch_end(trainer):
    """Log training metrics and model information at the end of an epoch."""
    if VERBOSE > 0:
        _log_plots(trainer.plots, step=trainer.epoch + 1)
        _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    if trainer.epoch == 0:
        wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)
    wb.run.log(trainer.metrics, step=trainer.epoch + 1, commit=True)  # commit forces sync


def on_train_epoch_end(trainer):
    """Log metrics and save images at the end of each training epoch."""
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix="train"), step=trainer.epoch + 1)
    wb.run.log(trainer.lr, step=trainer.epoch + 1)
    if VERBOSE > 0:
        if trainer.epoch == 1:
            _log_plots(trainer.plots, step=trainer.epoch + 1)


def on_train_end(trainer):
    """Save the best model as an artifact and log final plots at the end of training."""
    if VERBOSE > 0:
        _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
        _log_plots(trainer.plots, step=trainer.epoch + 1)
        # art = wb.Artifact(type="model", name=f"run_{wb.run.id}_model")
        # if trainer.best.exists():
        #     art.add_file(trainer.best)
        #     wb.run.log_artifact(art, aliases=["best"])
        # # Check if we actually have plots to save
        if trainer.args.plots and hasattr(trainer.validator.metrics, "curves_results"):
            for curve_name, curve_values in zip(trainer.validator.metrics.curves, trainer.validator.metrics.curves_results):
                x, y, x_title, y_title = curve_values
                _plot_curve(
                    x,
                    y,
                    names=list(trainer.validator.metrics.names.values()),
                    id=f"curves/{curve_name}",
                    title=curve_name,
                    x_title=x_title,
                    y_title=y_title,
                )
    wb.run.finish()  # required or run continues on dashboard


wb_callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if wb
    else {}
)
