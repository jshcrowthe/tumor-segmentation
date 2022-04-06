import os 
import torchio as tio
from tqdm import tqdm
from joblib import Parallel, delayed

class nni_utils:

    @staticmethod
    def get_folders(path):
        folders = os.listdir(path)
        folders = [folder for folder in folders if os.path.isdir(os.path.join(path,folder))]
        folders = sorted(folders,key = lambda x: int(x.split("_")[-1]))
        return folders
    
    @staticmethod
    def get_files(path,folder):

        files = sorted(os.listdir(os.path.join(path,folder)))
        files = [os.path.join(path,folder,file) for file in files]
        return files
    @staticmethod
    def create_downsample_record(dir,list,id):
        list = sorted(list)
        
        return tio.Subject(
                    label=tio.LabelMap(os.path.join(dir,list[1])),
                    data=tio.ScalarImage(os.path.join(dir,list[0])),
                    id = id
                )
        label=tio.LabelMap(os.path.join(dir,list[1]))
        data=tio.ScalarImage(os.path.join(dir,list[0]))
    @staticmethod
    def create_record(files,id, type):

        if type == "t1":

            return tio.Subject(
                    # flair=tio.ScalarImage(files[0],),
                    label=tio.LabelMap(files[1]),
                    t1=tio.ScalarImage(files[2],),
                    # t2=tio.ScalarImage(files[-1],)
                    id = id
                )
        if type == "t2":

            return tio.Subject(
                    # flair=tio.ScalarImage(files[0],),
                    label=tio.LabelMap(files[1]),
                    # t1=tio.ScalarImage(files[2],),
                    t2=tio.ScalarImage(files[-1],),
                    id = id
                )
        if type == "flair":

            return tio.Subject(
                    flair=tio.ScalarImage(files[0],),
                    label=tio.LabelMap(files[1]),
                    # t1=tio.ScalarImage(files[2],),
                    # t2=tio.ScalarImage(files[-1],)
                    id = id
                )
    @staticmethod
    def load_subjects(path,type):
        folders = nni_utils.get_folders(path)
        return [nni_utils.create_record(nni_utils.get_files(path,folder),id,type) for id,folder in enumerate(folders)]
    
    @staticmethod
    def load_downsampled_subjects(path):
        folders = nni_utils.get_folders(path)
        return [nni_utils.create_downsample_record(os.path.join(path,str(id)), nni_utils.get_files(path,folder),id) for id,folder in enumerate(folders)]

    @staticmethod
    def save_preprocessing(dir,seg,data,id):
        ending = ".nii.gz"
        path = os.path.join(dir,str(id))

        if not os.path.exists(path):
            os.mkdir(path)

        seg.save(os.path.join(dir,str(id),"seg"+ending))
        data.save(os.path.join(dir,str(id),"data"+ending))

    @staticmethod
    def downsample_preprocess(data_dir,out_dir,type,n_jobs =1):
        subjects = nni_utils.load_subjects(data_dir,type)

        size = (48, 60, 48)
        preprocessing_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(4),
            tio.CropOrPad(size),
        ])

        dataset = tio.SubjectsDataset(subjects, transform=preprocessing_transform)
        def func(i):
            seg,data,id = dataset[i]["label"],dataset[i][type],dataset[i]["id"]
            nni_utils.save_preprocessing(out_dir,seg,data,id)
        Parallel(n_jobs=n_jobs)(delayed(func)(i ) for i in tqdm(range(len(dataset))))
