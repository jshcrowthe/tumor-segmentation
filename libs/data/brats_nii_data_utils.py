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
    def create_record(files,id, type_list):

        type_list = type_list.copy()
        type_list.append("seg")
        records = {}

        for type in type_list:

            for file in files:

                if type + ".nii.gz"  in file:

                    if type == "seg":

                        records[type] = tio.LabelMap(file)

                    else:

                        records[type] = tio.ScalarImage(file)

        records["id"] = id

        return tio.Subject(records)

    @staticmethod
    def load_subjects(path,type_list):
        folders = nni_utils.get_folders(path)
        return [nni_utils.create_record(nni_utils.get_files(path,folder),id,type_list) for id,folder in enumerate(folders)]
    

    @staticmethod
    def save_preprocessing(dir,seg,data,id):
        ending = ".nii.gz"
        path = os.path.join(dir,str(id))

        if not os.path.exists(path):
            os.mkdir(path)

        seg.save(os.path.join(dir,str(id),"seg"+ending))
        for type,value in data.items():
            value.save(os.path.join(dir,str(id),type+ending))

    @staticmethod
    def downsample_preprocess(data_dir,out_dir,type_list,n_jobs =1):

        subjects = nni_utils.load_subjects(data_dir,type_list)

        size = (48, 60, 48)
        preprocessing_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(4),
            tio.CropOrPad(size),
        ])

        dataset = tio.SubjectsDataset(subjects, transform=preprocessing_transform)

        def func(i):
            
            seg = dataset[i]["seg"]
            id = dataset[i]["id"]
            data =  {type:dataset[i][type] for type in type_list}
            nni_utils.save_preprocessing(out_dir,seg,data,id)

        Parallel(n_jobs=n_jobs)(delayed(func)(i ) for i in tqdm(range(len(dataset))))
