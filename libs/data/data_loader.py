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
        train_transformer,
        val_transformer,
        data_dir = "/home/ludauter/Documents/brats_example/data/train",
        out_dir = "/home/ludauter/tumor-segmentation/libs/data/examples",
        type_list = ["t1"],
        sample_list = ["t1"],
        batch_size = 8,
        num_workers =16,
        prepare = False,
        n_jobs = 10,
        size = (48, 60, 48)
    ):
        super().__init__()
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.n_jobs = n_jobs
        self.prepare = prepare
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_loader = None
        self.validation_loader = None
        self.type_list = type_list
        self.size = (48, 60, 48)
        self.sample_list = sample_list
        self.train_transformer = train_transformer
        self.val_transformer = val_transformer
        self.size = size

    def prepare_data(self):
        if self.prepare:
            if not os.path.exists(self.out_dir):
                os.mkdir(self.out_dir)
            nni_utils.downsample_preprocess(self.data_dir,self.out_dir,self.sample_list,size = self.size,n_jobs = self.n_jobs)


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            subjects = nni_utils.load_subjects(self.out_dir,self.type_list)

            train_subjects, val_subjects = train_test_split(subjects,test_size = .2,random_state = 42 )

            training_set = tio.SubjectsDataset(train_subjects, transform=self.train_transformer)
            validation_set = tio.SubjectsDataset(val_subjects, transform=self.val_transformer)
    
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

        if stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.validation_loader