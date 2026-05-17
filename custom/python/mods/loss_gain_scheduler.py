class LossGainScheduler:

    def __init__(
            self,
            box: float = None,
            cls: float = None,
            dfl: float = None,
            cls_feat: float = None,
    ):
        self.box = box
        self.cls = cls
        self.dfl = dfl
        self.cls_feat = cls_feat

    def __call__(self, trainer):
        """
        for i, g in enumerate(trainer.optimizer.param_groups):
            print(f"Group {i} | lr={g['lr']:.6f}")
        """
        pass
