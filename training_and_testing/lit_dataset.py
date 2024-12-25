import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class inD_RecordingDataset(Dataset):
    def __init__(self, path, recording_id, sequence_length, features,  train=True):
        """Dataset for inD dataset.
        Parameters
        ----------
        path : str
            Path to the data.
        recording_id : int
            Recording id of the data.
        sequence_length : int
            Length of the sequence.
        features : list
            List of features to use.
        train : bool
            Whether to use the training set or not.
        """
        super(inD_RecordingDataset).__init__()
        self.path = path
        self.recording_id = recording_id
        self.sequence_length = sequence_length
        self.features = features
        self.train = train
        self.transform = self.get_transform()
        if type(self.recording_id) == list:
            self.data = pd.DataFrame()
            # TODO: Here we are simply loading the csv and stack them into one pandas dataframe.
            # You have to change this to load your data. This is just meant as a dummy example!!!
            for id in self.recording_id:
                with open(f"{path}/{id}_tracks.csv", 'rb') as f:
                    self.data = pd.concat([self.data, pd.read_csv(f, delimiter=',', header=0, usecols=self.features, dtype='float16')])
        else:
            with open(f"{path}/{recording_id}_tracks.csv", 'rb') as f:
                self.data = pd.read_csv(f, delimiter=',', header=0, usecols=self.features, dtype='float16')


    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """
                 Returns the item at index idx.
        Parameters
        ----------
        idx : int
            Index of the item.
        Returns
        -------
        data : torch.Tensor
            The data at index idx.
        """
        if idx <= self.__len__():
            data = self.data[idx:idx + self.sequence_length]

            if self.transform:
                data = self.transform(np.array(data, dtype='float16')).squeeze()
            return data
        else:
            print("wrong index")
            return None

    def get_transform(self):
        """
        Returns the transform for the data.
        """
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        return data_transforms