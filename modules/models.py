import logging
from torch import nn
import pytorch_lightning as pl
import torchmetrics

class Model(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizers, lr_schedulers):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self._optimizers = optimizers
        self._lr_schedulers = lr_schedulers
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        
        self.log('train/loss', loss, prog_bar=False, logger=True)
        self.log('train/acc', self.train_acc(pred,y), prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log('val/acc', self.val_acc(pred,y), prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log('test/acc', self.test_acc(pred,y), prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return {
        "optimizer": self._optimizers,
        "lr_scheduler": {
            "scheduler": self._lr_schedulers,
            "interval": "step",
            "frequency": 100,
        },
    }
