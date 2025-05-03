import torch
import pytorch_lightning as pl
import wandb

class ResolutionTestCallback(pl.Callback):
    def __init__(self, resolution: int, is_last: bool = False):
        """
        resolution: the scalar resolution (e.g. 128) you want to test at.
        """
        super().__init__()

        self.resolution = int(20.0/resolution)
        self._orig_ds_res = None
        self._orig_comment_log = None
        self.is_last = is_last
        self.table = wandb.Table(columns=["resolution", "accuracy", "mAP"])

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # 1) Tag all your logs for this run:
        self._orig_comment_log = pl_module.comment_log
        pl_module.comment_log = f"{self.resolution}x{self.resolution} "+pl_module.comment_log

        # 2) Reset the per-class test metrics so they don't carry over
        pl_module.metric_test_accuracy_per_class.reset()
        pl_module.metric_test_AP_per_class.reset()

    def on_test_epoch_end(self):
        pass


    def on_test_end(self, trainer, pl_module):
        # compute just‐finished resolution metrics
        per_class_acc = pl_module.metric_test_accuracy_per_class.compute() * 100
        per_class_ap  = pl_module.metric_test_AP_per_class.compute() * 100
        acc, mean_ap  = per_class_acc.mean().item(), per_class_ap.mean().item()

        # 1) add to our table
        self.table.add_data(self.resolution, acc, mean_ap)

        # 2) if this is the last resolution in your sweep, log the scatter
        if self.is_last:
            run = trainer.logger.experiment   # this is your wandb.run
            scatter_acc = wandb.plot.scatter(
                self.table, "resolution", "accuracy",
                title=pl_module.comment_log+" Resolution vs Test Accuracy"
            )
            scatter_map = wandb.plot.scatter(
                self.table, "resolution", "mAP",
                title="Resolution vs Test mAP"
            )
            run.log({
                "resolution_vs_accuracy": scatter_acc,
                "resolution_vs_mAP"     : scatter_map,
                "resolution_table"      : self.table
            })
