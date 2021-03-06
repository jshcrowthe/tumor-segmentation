import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from metrics.metrics import accuracy, dice_coef, iou
from models import (
    OneLayer,
    PixelModel,
    Unet,
    DeepLabV3Plus,
    CustomModel,
)
from losses.losses import (
    CrossEntropy,
    FocalTverskyLoss,
    DiceLoss,
    FocalLoss,
    LogCoshLoss,
    JacardLoss,
)

class Main_Loop(pl.LightningModule):
    def __init__(
        self,
        model="OneLayer",
        loss="CrossEntropy",
        optimizer=torch.optim.AdamW,
        type_list=["t1"],
        learning_rate=0.01,
        num_classes=2,
        batch_size=8,
        scheduler=None,
        scheduler_args={},
        loss_args={},
        model_args={},
        optimizer_args={},
    ):
        super().__init__()
        self.model = self.get_model(model)(**model_args)
        self.learning_rate = learning_rate
        self.loss = self.get_loss(loss)(**loss_args)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args
        self.type_list = type_list
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.save_hyperparameters()

    def get_model(self, model):

        if model == "OneLayer":
            return OneLayer
        elif model == "Pixel":
            return PixelModel
        elif model == "Unet":
            return Unet
        elif model == "deeplabv3plus":
            return DeepLabV3Plus
        elif model == "custom":
            return CustomModel
        else:
            return "Spelling Mistake"

    def get_loss(self, loss):

        if loss == "CrossEntropy":
            return CrossEntropy
        elif loss == "Dice":
            return DiceLoss
        elif loss == "Jacard":
            return JacardLoss
        elif loss == "LogCosh":
            return LogCoshLoss
        elif loss == "Focal":
            return FocalLoss
        elif loss == "Tversky":
            return FocalTverskyLoss
        else:
            return "Spelling Mistake"

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), **self.optimizer_args)
        if self.scheduler:
            lr_scheduler = {
                "scheduler": self.scheduler(optimizer=optimizer, **self.scheduler_args),
                "interval": "epoch",
                "strict": True,
            }

            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def prepare_batch(self, batch):
        if len(self.type_list) == 1:
            inputs = batch[self.type_list[0]]["data"]
            targets = batch["seg"]["data"].long()
            return inputs, targets
        else:
            targets = batch["seg"]["data"].long()
            n = len(self.type_list)
            inputs = 0
            for type in self.type_list:
                inputs = inputs + batch[type]["data"]
            inputs = inputs / n
            return inputs, targets

    def training_step(self, batch, batch_idx):

        inputs, targets = self.prepare_batch(batch)

        logits = self.model(inputs)
        batch_loss = self.loss(logits, targets)

        pred = logits.argmax(dim=1)
        with torch.no_grad():
            self.log(
                "Train Loss",
                float(batch_loss.cpu().numpy()),
                on_epoch=True,
                on_step=True,
                batch_size=self.batch_size,
            )
            self.log(
                "Train Acc",
                float(accuracy(pred, targets).cpu().numpy()),
                on_epoch=True,
                on_step=True,
                batch_size=self.batch_size,
            )
            self.log(
                "Train mIoU",
                float(iou(logits, targets).cpu().numpy()),
                on_epoch=True,
                on_step=True,
                batch_size=self.batch_size,
            )
            self.log(
                "Train dice_coef",
                float(dice_coef(logits, targets).cpu().numpy()),
                on_epoch=True,
                on_step=True,
                batch_size=self.batch_size,
            )

        return batch_loss

    def test_step(self, batch, batch_idx):

        inputs, targets = self.prepare_batch(batch)

        logits = self.model(inputs)
        batch_loss = self.loss(logits, targets)

        pred = logits.argmax(dim=1)
        with torch.no_grad():
            self.log(
                "Test Loss",
                float(batch_loss.cpu().numpy()),
                on_epoch=True,
                on_step=True,
                batch_size=self.batch_size,
            )
            self.log(
                "Test Acc",
                float(accuracy(pred, targets).cpu().numpy()),
                on_epoch=True,
                on_step=True,
                batch_size=self.batch_size,
            )
            self.log(
                "Test mIoU",
                float(iou(logits, targets).cpu().numpy()),
                on_epoch=True,
                on_step=True,
                batch_size=self.batch_size,
            )
            self.log(
                "Test dice_coef",
                float(dice_coef(logits, targets).cpu().numpy()),
                on_epoch=True,
                on_step=True,
                batch_size=self.batch_size,
            )

        return batch_loss

    def validation_step(self, batch, batch_idx):

        inputs, targets = self.prepare_batch(batch)

        logits = self.model(inputs)

        batch_loss = self.loss(logits, targets)

        pred = logits.argmax(dim=1)

        with torch.no_grad():

            self.log(
                "Validation Loss",
                float(batch_loss.cpu().numpy()),
                on_epoch=True,
                on_step=True,
                batch_size=self.batch_size,
            )

            self.log(
                "Validation Acc",
                float(accuracy(pred, targets).cpu().numpy()),
                on_epoch=True,
                on_step=True,
                batch_size=self.batch_size,
            )
            self.log(
                "Validation mIoU",
                float(iou(logits, targets).cpu().numpy()),
                on_epoch=True,
                on_step=True,
                batch_size=self.batch_size,
            )
            self.log(
                "Validation dice_coef",
                float(dice_coef(logits, targets).cpu().numpy()),
                on_epoch=True,
                on_step=True,
                batch_size=self.batch_size,
            )

        return batch_loss
