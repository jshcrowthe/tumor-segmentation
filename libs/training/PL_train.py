import pytorch_lightning as pl
# import numpy as np
import torch
import torch.nn.functional as F

class Main_Loop(pl.LightningModule):

    def __init__(self, model,loss,metric,learning_rate=0.01,num_classes = 2,batch_size = 8,scheduler = None,scheduler_args = None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = loss
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.metric = metric
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args
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

    def training_step(self, batch, batch_idx):

        inputs = batch["t1"]["data"]
        targets = batch["seg"]["data"].long()

        logits = self.model(inputs)
        batch_loss = self.loss(logits, targets)

        pred = logits.argmax(dim = 1)
        with torch.no_grad():
            self.log("Train Loss",float(batch_loss.cpu().numpy()) , on_epoch=True,on_step = True,batch_size=self.batch_size)
            self.log("Train Acc",float(self.metric(pred,targets).cpu().numpy()) , on_epoch=True,on_step = True,batch_size=self.batch_size)
        
        return batch_loss

    def validation_step(self, batch, batch_idx):

        inputs = batch["t1"]["data"]
        targets = batch["seg"]["data"].long()

        logits = self.model(inputs)

        batch_loss = self.loss(logits, targets)

        pred = logits.argmax(dim = 1)

        with torch.no_grad():
 
            self.log("Validation Loss",float(batch_loss.cpu().numpy()) , on_epoch=True,on_step = True,batch_size=self.batch_size)

            self.log("Validation Acc",float(self.metric(pred,targets).cpu().numpy()) , on_epoch=True,on_step = True,batch_size=self.batch_size)

        return batch_loss
