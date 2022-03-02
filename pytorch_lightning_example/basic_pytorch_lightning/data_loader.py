import pytorch_lightning as pl
from pathlib import Path
import os 
import torchio as tio
import torch

class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/VOCtrainval_11-May-2012",
        batch_size: int = 8,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_loader = None
        self.validation_loader = None
        self.training_set = None
        self.validation_set = None

    def prepare_data(self):
        training_split_ratio = 0.9 
        dataset_url = 'https://www.dropbox.com/s/ogxjwjxdv5mieah/ixi_tiny.zip?dl=0'
        dataset_path = 'ixi_tiny.zip'
        dataset_dir_name = 'ixi_tiny'
        dataset_dir = Path(dataset_dir_name)
        histogram_landmarks_path = 'landmarks.npy'
        if not dataset_dir.is_dir():
            os.system("curl --silent --output {0} --location {1}".format(dataset_path,dataset_url) )
            os.system("unzip -qq {}".format(dataset_path))

        images_dir = dataset_dir / 'image'
        labels_dir = dataset_dir / 'label'
        image_paths = sorted(images_dir.glob('*.nii.gz'))
        label_paths = sorted(labels_dir.glob('*.nii.gz'))
        
        landmarks = tio.HistogramStandardization.train(
            image_paths,
            output_path=histogram_landmarks_path,
        )
        subjects = []
        for (image_path, label_path) in zip(image_paths, label_paths):
            subject = tio.Subject(
                mri=tio.ScalarImage(image_path),
                brain=tio.LabelMap(label_path),
            )
            subjects.append(subject)
        dataset = tio.SubjectsDataset(subjects)

        training_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(4),
            tio.CropOrPad((48, 60, 48)),
            tio.RandomMotion(p=0.2),
            tio.HistogramStandardization({'mri': landmarks}),
            tio.RandomBiasField(p=0.3),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.RandomNoise(p=0.5),
            tio.RandomFlip(),
            tio.OneOf({
                tio.RandomAffine(): 0.8,
                tio.RandomElasticDeformation(): 0.2,
            }),
            tio.OneHot(),
        ])

        validation_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(4),
            tio.CropOrPad((48, 60, 48)),
            tio.HistogramStandardization({'mri': landmarks}),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.OneHot(),
        ])

        num_subjects = len(dataset)
        num_training_subjects = int(training_split_ratio * num_subjects)
        num_validation_subjects = num_subjects - num_training_subjects

        num_split_subjects = num_training_subjects, num_validation_subjects
        training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

        self.training_set = tio.SubjectsDataset(
        training_subjects, transform=training_transform)

        self.validation_set = tio.SubjectsDataset(
            validation_subjects, transform=validation_transform)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            training_batch_size = 16
            validation_batch_size = 1
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