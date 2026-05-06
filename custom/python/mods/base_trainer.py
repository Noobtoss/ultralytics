from ultralytics.utils import RANK, LOCAL_RANK, LOGGER
from ultralytics.utils.torch_utils import strip_optimizer, torch_distributed_zero_first
from ultralytics.engine.trainer import BaseTrainer as BaseBaseTrainer


class BaseTrainer(BaseBaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("Modded BaseTrainer __init__ called")
        super().__init__(*args, **kwargs)

    def final_eval(self):
        """Perform final evaluation and validation for the YOLO model."""
        # model = self.best if self.best.exists() else None
        model = self.last if self.last.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):  # strip only on GPU 0; other GPUs should wait
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    # update best.pt train_metrics from last.pt
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        if model:
            LOGGER.info(f"\nValidating {model}...")
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False  # disable final val compile as too slow
            self.metrics = self.validator(model=model)
            self.metrics.pop("fitness", None)
            self.run_callbacks("on_fit_epoch_end")
