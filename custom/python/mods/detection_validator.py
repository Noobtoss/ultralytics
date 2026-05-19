from pathlib import Path
from typing import Any
import numpy as np
from ultralytics.engine.validator import BaseValidator as _BaseValidator
from ultralytics.models.yolo.detect import DetectionValidator as _DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import DetMetrics


class BaseValidator(_BaseValidator):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] BaseValidator")
        super().__init__(*args, **kwargs)


_DetectionValidator.__bases__ = (BaseValidator,)


class DetectionValidator(_DetectionValidator):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] DetectionValidator")
        super().__init__(*args, **kwargs)
        self.metrics_agnostic = DetMetrics(names={0: "agnostic"})
        self.mAP_subsets = {k: v for k, v in (getattr(self.args, "mAP_subsets", None) or {}).items()}

    def init_metrics(self, model) -> None:
        super().init_metrics(model)

        if not self.mAP_subsets:
            all_classes = list(range(self.nc))
            self.mAP_subsets = {
                "No0": all_classes[1:],
                "New": all_classes[21:],
            }

    def update_metrics(self, preds, batch):
        # normal metrics — already handled by super()
        super().update_metrics(preds, batch)

        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)
            pbatch = {**pbatch, "cls": pbatch["cls"] * 0}
            predn = {**predn, "cls": predn["cls"] * 0}

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0

            self.metrics_agnostic.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                    "im_name": Path(pbatch["im_file"]).name,
                }
            )

    def get_stats(self) -> dict[str, Any]:
        """Calculate and return metrics statistics.

        Returns:
            (dict[str, Any]): Dictionary containing metrics results.
        """
        metrics = super().get_stats()

        # --------------------------------------------------------------------------------------------------------------

        self.metrics_agnostic.process(save_dir=self.save_dir, plot=False, on_plot=self.on_plot)
        self.metrics_agnostic.clear_stats()
        metrics_agnostic = self.metrics_agnostic.results_dict
        metrics["metrics/mAP50-95(B)_agnostic"] = metrics_agnostic["metrics/mAP50-95(B)"]

        # --------------------------------------------------------------------------------------------------------------

        ap_class_index = self.metrics.ap_class_index
        ap_values = self.metrics.box.ap  # (nc_present, 10)

        for name, subset in self.mAP_subsets.items():
            keep = np.array([ci in subset for ci in ap_class_index])
            if keep.any():
                metrics[f"metrics/mAP50-95(B)_{name}"] = float(ap_values[keep].mean())
            else:
                metrics[f"metrics/mAP50-95(B)_{name}"] = 0.0

        # --------------------------------------------------------------------------------------------------------------

        metrics = {k.replace("(B)", "").replace("metrics/", ""): v for k, v in metrics.items()}

        return metrics

    def print_results(self) -> None:

        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format

        # --------------------------------------------------------------------------------------------------------------

        LOGGER.info(pf % (
            "agnostic", self.seen, self.metrics_agnostic.nt_per_class.sum(), *self.metrics_agnostic.mean_results()))

        # --------------------------------------------------------------------------------------------------------------

        for name, subset in self.mAP_subsets.items():
            ap_class_index = self.metrics.ap_class_index
            ap_values = self.metrics.box.ap
            ap50_values = self.metrics.box.ap50
            p_values = self.metrics.box.p  # (nc_present,)
            r_values = self.metrics.box.r  # (nc_present,)

            keep = np.array([ci in subset for ci in ap_class_index])
            nt_subset = self.metrics.nt_per_class[list(subset)].sum()

            if keep.any():
                p = float(p_values[keep].mean())
                r = float(r_values[keep].mean())
                map50 = float(ap50_values[keep].mean())
                map50_95 = float(ap_values[keep].mean())
            else:
                p = r = map50 = map50_95 = 0.0

            LOGGER.info(pf % (name, self.seen, nt_subset, p, r, map50, map50_95))

        # --------------------------------------------------------------------------------------------------------------

        super().print_results()
