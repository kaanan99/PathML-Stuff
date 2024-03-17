import nvidia.dali.fn as fn
import nvidia.dali.types as types
import os
import pytorch_lightning as pl

from config import *
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator


@pipeline_def 
def video_pipe(file_list):
    videos, labels = fn.readers.video(
        device="gpu", 
        sequence_length=SEQUENCE_LENGTH, 
        random_shuffle=True, 
        initial_fill=2048, # Ask what this means
        file_list_include_preceding_frame=False,
        dtype=types.FLOAT, 
        file_list_frame_num=True, 
        file_list=file_list
    )

    # H for most images is 189, so pad smaller images to match that, all widths are 224 so no need to pad
    videos = fn.pad(videos, axes=[1], shape=189)

    # Nomalize image (I think this is using image mean and std, but what I want to do is actually divide by 255)
    # videos = fn.normalize(videos)

    videos = videos / 255
    
    # One hot encode labels
    labels = fn.one_hot(labels, num_classes=NUM_CLASSES)
    
    return videos, labels


@pipeline_def
def test_video_pipe(file_list):
    videos, labels = fn.readers.video(
        device="gpu", 
        sequence_length=SEQUENCE_LENGTH, 
        random_shuffle=True,                   
        initial_fill=256, 
        file_list_include_preceding_frame=False,                
        dtype=types.FLOAT, 
        file_list_frame_num=True, 
        file_list=file_list
    )

    # H for most images is 189, so pad smaller images to match that, all widths are 224 so no need to pad
    videos = fn.pad(videos, axes=[1], shape=189)

     # Nomalize image (I think this is using image mean and std, but what I want to do is actually divide by 255)
    # videos = fn.normalize(videos)

    videos = videos / 255

    # One hot encode labels
    labels = fn.one_hot(labels, num_classes=NUM_CLASSES)    
    
    return videos, labels


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, training_file_path, testing_file_path, eval_file_path, train_size, test_size, eval_size):
        super().__init__()
        self.training_path = training_file_path
        self.testing_path = testing_file_path
        self.eval_path = eval_file_path
        self.train_size = train_size
        self.test_size = test_size
        self.eval_size = eval_size
        self.batch_size = batch_size

        # Move to root directory
        cwd = os.getcwd()
        root = os.path.abspath(os.sep)
        os.chdir(root)

        # Create pipes
        self.training_pipe = video_pipe(
            batch_size=self.batch_size, 
            num_threads=3, 
            device_id=0, 
            file_list=self.training_path, 
            seed=123456
        )
        self.test_pipe = test_video_pipe(
            batch_size=VAL_BATCH_SIZE, 
            num_threads=1, 
            device_id=0, 
            file_list=self.testing_path, 
            seed=123456
        )
        self.eval_pipe = test_video_pipe(
            batch_size=VAL_BATCH_SIZE, 
            num_threads=1, 
            device_id=0, 
            file_list=self.eval_path, 
            seed=123456
        )

        # Build pipies
        self.training_pipe.build()
        self.test_pipe.build()
        self.eval_pipe.build()

        # Change back to current directory
        os.chdir(cwd)


    def train_dataloader(self):
        return DALIClassificationIterator(self.training_pipe, size=self.train_size)
        

    # "validation" in PyTorch is run each epoch (Testing)
    def val_dataloader(self):
        return DALIClassificationIterator(self.test_pipe, size=self.test_size)


    # "test" in PyTorch is run at the end (Evaluation)
    def test_dataloader(self):
        return DALIClassificationIterator(self.eval_pipe, size=self.eval_size)
