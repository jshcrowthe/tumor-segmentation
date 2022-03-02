import pytorch_lightning as pl
from utils import  prepare_batch , get_dice_loss  , AverageMeter
# import numpy as np
from unet import UNet
import torch
import torch.nn.functional as F

class Model(pl.LightningModule):

    def __init__(self, learning_rate=0.01,num_classes = 2,epochs = 10):
        super().__init__()
        self.model = UNet(
        in_channels=1,
        out_classes=2,
        dimensions=3,
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        normalization='batch',
        upsampling_type='linear',
        padding=True,
        activation='PReLU',
    )
        self.learning_rate = learning_rate
        self.loss = ""
        self.num_classes = num_classes
        self.metrics = {"Dice_Loss"}
        self.train_Stats = AverageMeter()
        self.validation_Stats = AverageMeter()
        self.epochs = epochs

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        return optimizer#,[lr_scheduler]

    def training_step(self, batch, batch_idx):

        inputs, targets = prepare_batch(batch)
        logits = self.model(inputs)
        probabilities = F.softmax(logits, dim=1)
        batch_losses = get_dice_loss(probabilities, targets)
        batch_loss = batch_losses.mean()

        with torch.no_grad():
            self.log("Train Loss",float(batch_loss.cpu().numpy()) , on_epoch=True,on_step = True,batch_size=16)
        
        return batch_loss

    def on_validation_epoch_start(self):
        self.validation_Stats.reset()

    def validation_step(self, batch, batch_idx):

        inputs, targets = prepare_batch(batch)
        logits = self.model(inputs)
        probabilities = F.softmax(logits, dim=1)
        batch_losses = get_dice_loss(probabilities, targets)
        batch_loss = batch_losses.mean()

        with torch.no_grad():
            self.log("Validation Loss",float(batch_loss.cpu().numpy()) , on_epoch=True,on_step = True,batch_size=16)

        return batch_loss



