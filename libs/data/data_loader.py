import pytorch_lightning as pl
from pathlib import Path
import os 
import torchio as tio
import torch
from .brats_nii_data_utils import *
from sklearn.model_selection import train_test_split

class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir = "/home/ludauter/Documents/brats_example/data/train",
        out_dir = "/home/ludauter/tumor-segmentation/libs/data/examples",
        data_type = "t1",
        batch_size = 8,
        num_workers =10,
        prepaire = False,
        n_jobs = 10
    ):
        super().__init__()
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.n_jobs = n_jobs
        self.prepaire = prepaire
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_loader = None
        self.validation_loader = None
        self.data_type = data_type
        self.size = (48, 60, 48)

    def prepare_data(self):
        if self.prepaire:
            nni_utils.downsample_preprocess(self.data_dir,self.out_dir,self.data_type,n_jobs = 10)
        subjects = nni_utils.load_downsampled_subjects(self.out_dir)
        train_subjects, val_subjects = train_test_split(subjects,test_size = .2,random_state = 42 )
        size = (48, 60, 48)
        training_transform = tio.Compose([
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
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        ])

        training_set = tio.SubjectsDataset(train_subjects, transform=training_transform)
        validation_set = tio.SubjectsDataset(val_subjects, transform=validation_transform)
 
        self.train_loader = torch.utils.data.DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        self.validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            pass

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.validation_loader


    