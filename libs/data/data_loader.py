import pytorch_lightning as pl
from pathlib import Path
import os 
import torchio as tio
import torch
from .brats_nii_data_utils import *

class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir = "/home/ludauter/Documents/brats_example/brats_train",
        data_type = "t1",
        batch_size = 4,
        num_workers =5,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_loader = None
        self.validation_loader = None
        self.training_set = None
        self.validation_set = None
        self.data_type = data_type

    def prepare_data(self):
        subjects = nni_utils.load_subjects(self.data_dir,self.data_type)
        num_subjects = len(subjects)
        num_split_subjects = [int(num_subjects*.8),num_subjects-int(num_subjects*.8)]
        train_subjects, val_subjects = torch.utils.data.random_split(subjects,num_split_subjects,generator=torch.Generator().manual_seed(42) )
        size = (48, 60, 48)
        training_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(4),
            tio.CropOrPad(size),
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

        validation_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(1),
            tio.CropOrPad(size),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        ])

        self.training_set = tio.SubjectsDataset(train_subjects, transform=training_transform)
        self.validation_set = tio.SubjectsDataset(val_subjects, transform=validation_transform)


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            training_batch_size = self.batch_size
            validation_batch_size = self.batch_size
            self.train_loader = torch.utils.data.DataLoader(
                self.training_set,
                batch_size=training_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

            self.validation_loader = torch.utils.data.DataLoader(
                self.validation_set,
                batch_size=validation_batch_size,
                num_workers=self.num_workers,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.validation_loader