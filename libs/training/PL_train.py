import pytorch_lightning as pl
# import numpy as np
import torch
import torch.nn.functional as F
from metrics.accuracy  import accuracy
from metrics.dice_coef import dice_coef
from metrics.iou import iou

class Main_Loop(pl.LightningModule):

    def __init__(self, model,loss,type_list,learning_rate=0.01,num_classes = 2,batch_size = 8,scheduler = None,scheduler_args = None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = loss
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args
        self.type_list = type_list
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        if self.scheduler:
            lr_scheduler = {
            'scheduler': self.scheduler(optimizer = optimizer,**self.scheduler_args),
            'interval': 'epoch',
            'strict': True
            }

            return [optimizer],[lr_scheduler]
        else:
            return optimizer

    def prepare_batch(self,batch):
        if len(self.type_list) == 1:
            inputs = batch[self.type_list[0]]["data"]
            targets = batch["seg"]["data"].long()
            return inputs,targets
        else:
            targets = batch["seg"]["data"].long()
            n = len(self.type_list)
            inputs = 0
            for type in self.type_list:
                inputs = inputs + batch[type]["data"]
            inputs = inputs/n
            return inputs,targets
            
    def training_step(self, batch, batch_idx):

        inputs,targets = self.prepare_batch(batch)

        logits = self.model(inputs)
        batch_loss = self.loss(logits, targets)

        pred = logits.argmax(dim = 1)
        with torch.no_grad():
            self.log("Train Loss",float(batch_loss.cpu().numpy()) , on_epoch=True,on_step = True,batch_size=self.batch_size)
            self.log("Train Acc",float(accuracy(pred,targets).cpu().numpy()) , on_epoch=True,on_step = True,batch_size=self.batch_size)
            self.log("Train mIoU",float(iou(logits,targets).cpu().numpy()) , on_epoch=True,on_step = True,batch_size=self.batch_size)
            self.log("Train dice_coef",float(dice_coef(logits,targets).cpu().numpy()) , on_epoch=True,on_step = True,batch_size=self.batch_size)
        
        return batch_loss

    def validation_step(self, batch, batch_idx):

        inputs,targets = self.prepare_batch(batch)

        logits = self.model(inputs)

        batch_loss = self.loss(logits, targets)

        pred = logits.argmax(dim = 1)

        with torch.no_grad():
 
            self.log("Validation Loss",float(batch_loss.cpu().numpy()) , on_epoch=True,on_step = True,batch_size=self.batch_size)

            self.log("Validation Acc",float(accuracy(pred,targets).cpu().numpy()) , on_epoch=True,on_step = True,batch_size=self.batch_size)
            self.log("Validation mIoU",float(iou(logits,targets).cpu().numpy()) , on_epoch=True,on_step = True,batch_size=self.batch_size)
            self.log("Validation dice_coef",float(dice_coef(logits,targets).cpu().numpy()) , on_epoch=True,on_step = True,batch_size=self.batch_size)
        
        return batch_loss
