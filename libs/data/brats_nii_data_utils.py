import os 
import torchio as tio


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