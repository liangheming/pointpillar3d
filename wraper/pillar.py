import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from models.pointpillars import PointPillar
from utils.general import ModelEMA


class PillarWrapper(LightningModule):
    def __init__(self, hyparams):
        super().__init__()
        self.pillar = PointPillar(n_classes=3)
        self.ema = ModelEMA(self.pillar)

    def on_train_start(self):
        self.ema.ema.to(self.device)

    def training_step(self, batch, batch_idx):
        v, c, pn, an, ca = batch
        v.to(self.device)
        c.to(self.device)
        pn.to(self.device)
        ret = self.pillar(v, c, pn, an)
        for k, v in ret.items():
            self.log("train/{:s}".format(k), v.float().item())
        self.log("loss", ret['total_loss'].item(), prog_bar=True, on_step=True, batch_size=6)
        return ret['total_loss']

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update(self.pillar, decay=None)

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches
        optimizer = AdamW(self.pillar.parameters(),
                          lr=0.00025,
                          betas=(0.95, 0.99),
                          weight_decay=0.01)

        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.0025,
            epochs=160,
            steps_per_epoch=self.trainer.num_training_batches,
            pct_start=0.4,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.95 * 0.895,
            max_momentum=0.85,
            div_factor=10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def validation_step(self, batch, batch_idx):
        v, c, pn, an, ca = batch
        v.to(self.device)
        c.to(self.device)
        pn.to(self.device)
        boxes = self.pillar(v, c, pn, an)['boxes']
        nums = [len(b) for b in boxes if b is not None]
        nums.append(0)
        num = sum(nums)
        self.log("val/num", torch.tensor(num).float(), prog_bar=True, batch_size=6)

    def on_validation_epoch_end(self):
        self.log("val_epoch", torch.tensor(self.current_epoch).float())

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.ema.state_dict()
