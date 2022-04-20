import pytorch_lightning as pl
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
        data_dir,
        out_dir,
        type_list = ["t1"],
        sample_list = ["t1"],
        batch_size = 4,
        num_workers =5,
        n_jobs = 10,
        size = (48, 60, 48)
    ):
        super().__init__()
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        self.type_list = type_list
        self.size = (48, 60, 48)
        self.sample_list = sample_list
        self.train_transformer = train_transformer
        self.val_transformer = val_transformer
        self.size = size
        self.valid_dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def preprocessing(self):
        subjects = nni_utils.load_subjects(self.data_dir,self.type_list)
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        nni_utils.downsample_preprocess(subjects,self.out_dir,self.sample_list,size = self.size,n_jobs = self.n_jobs)


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            subjects = nni_utils.load_subjects(self.out_dir,self.type_list)

            train_subjects, other_subjects = train_test_split(subjects,train_size = 800,random_state = 42 )
            val_subjects, test_subjects = train_test_split(subjects,train_size = 200,random_state = 42 )


            self.train_dataset = tio.SubjectsDataset(train_subjects, transform=self.train_transformer)
            self.valid_dataset = tio.SubjectsDataset(val_subjects, transform=self.val_transformer)
            self.test_dataset = tio.SubjectsDataset(test_subjects, transform=self.val_transformer)


            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset ,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

            self.validation_loader = torch.utils.data.DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )


            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
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

    def test_dataloader(self):
        return self.test_loader
