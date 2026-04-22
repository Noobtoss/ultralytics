from ultralytics.models.yolo.detect import DetectionTrainer

from .train_loss import TrainLoss


# THS, Copied from ultralytics.models.yolo.detect.DetectionTrainer
# THS, Copied from ultralytics.engine.trainer.BaseTrainer

class Trainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        return model

    def init_criterion(self):
        """Replace default loss with your custom one."""
        return TrainLoss(self.model)
