import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from wraper.pillar import PillarWrapper
from wraper.data import KittiWrapper
from pytorch_lightning.loggers import TensorBoardLogger

pl.seed_everything(128)
torch.set_float32_matmul_precision("high")


class ProgressBar(TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    system = PillarWrapper(None)
    dm = KittiWrapper(None)
    logger = TensorBoardLogger(
        save_dir="workspace",
        name="pillar",
        default_hp_metric=False
    )
    bar_callback = ProgressBar(refresh_rate=5)
    ckpt_dir = "workspace/{:s}/version_{:d}/ckpt".format("pillar",
                                                         logger.version)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                          filename='{epoch}-{val_epoch:.4f}',
                                          monitor='val_epoch',
                                          mode='max',
                                          save_last=True,
                                          save_weights_only=True,
                                          save_top_k=5)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        default_root_dir=ckpt_dir,
        accelerator='gpu',
        devices=[0],
        max_epochs=160,
        num_sanity_val_steps=0,
        callbacks=[bar_callback, checkpoint_callback, lr_monitor],
        logger=logger,
        benchmark=True,
        check_val_every_n_epoch=10,
    )
    trainer.fit(system, dm)
