import os
import sys
import tarfile

# Patch the path to include local libs
sys.path.insert(0, os.path.abspath("./libs"))

# Import
import pytorch_lightning as pl
from data.data_loader import MyDataModule
from training.PL_train import Main_Loop
import torch.optim as optim
import torchio as tio
import torch
import numpy as np

PROCESSED_DATA_PATH = os.path.abspath("./data/processed")
RAW_DATA_PATH = os.path.abspath("./data/raw")

size = (48, 64, 48)
model = input("Enter model to train: ")

batch_size = 4
epochs = 50
type_list = ['t1', 't2', 'flair']
weight = torch.from_numpy(np.array([0.1, 1, 1, 1, 1])).float().cuda()
model_args = {}

# Experiments
losses = ['Focal', 'LogCosh', 'Dice']
lr_schedules = [
  optim.lr_scheduler.ExponentialLR,
  optim.lr_scheduler.CosineAnnealingLR,
  optim.lr_scheduler.StepLR,
]

scheduler_args = [
  {"gamma": 0.95},
  {"T_max":10},
  {"step_size":5}
]

optimizers = [
  optim.Adam,
  optim.RMSprop,
  optim.Adagrad
]

optimizer_args = [
  {"amsgrad": True},
  {"centered":True},
  {}
]

# Data TRansforms
train_transformer = tio.Compose([
  tio.RandomMotion(p=0.2),
  tio.RandomBiasField(p=0.3),
  tio.ZNormalization(masking_method=tio.ZNormalization.mean),
  tio.RandomNoise(p=0.5),
  tio.RandomFlip(),
  tio.OneOf({
    tio.RandomAffine(): 0.8,
    tio.RandomElasticDeformation(): 0.2,
  }),
])

val_transformer = tio.Compose([
  tio.ZNormalization(masking_method=tio.ZNormalization.mean),
])

# Dataloading
data_module = MyDataModule(
  data_dir=RAW_DATA_PATH,
  out_dir=PROCESSED_DATA_PATH,
  train_transformer=train_transformer,
  val_transformer=val_transformer,
  size=size,
  type_list=type_list,
  sample_list=type_list,
)

if len(os.listdir(RAW_DATA_PATH)) <= 1:
    tarball_path = input("Path to BRATS 2021 training tarball")
    tarball_path = os.path.abspath(tarball_path)

    if tarfile.is_tarfile(tarball_path):
        # open file
        file = tarfile.open(tarball_path)

        # extracting file
        file.extractall(RAW_DATA_PATH)

        file.close()
    else:
        raise Exception("Valid tarball path not passed")

if len(os.listdir(PROCESSED_DATA_PATH)) <= 1:
    print("Processing images")
    data_module.preprocessing()

for loss in losses:
  for lr_schedule,s_args in zip(lr_schedules,scheduler_args):
    for optimizer,o_args in zip(optimizers,optimizer_args):
      for type in type_list:
        trainer = pl.Trainer(gpus=1, max_epochs=epochs)
        main = Main_Loop(
          model=model,
          loss=loss,
          type_list=[type],
          scheduler=lr_schedule,
          scheduler_args=s_args,
          model_args=model_args,
          loss_args={"weight": weight},
          batch_size=batch_size,
          optimizer=optimizer,
          optimizer_args=o_args,
        )
        trainer.fit(main, data_module)
        trainer.test(main, data_module)