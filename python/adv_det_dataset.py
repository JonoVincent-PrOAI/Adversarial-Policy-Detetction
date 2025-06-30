import torch 
from torch.utils.data import Dataset
import os
import numpy

class Adversarial_Detection_Dataset(Dataset):

    def __init__(self, dir_path : str):
        self.dir_path = dir_path
        self.data_dir = os.fsencode(dir_path)
        self.game_length_index = []
        for game_file in os.listdir(self.data_dir):
            game_file = dir_path + '/' + os.fsdecode(game_file)
            with open(os.fsdecode(game_file), 'rb') as f:
                game_data = np.load(game_file, allow_pickle=True)
                self.game_length_index.append(len(game_data))

    def __getitem__(self, idx):
        game_count = 0
        game_index = -1    
        while idx > game_count:
            game_index = game_index + 1
            game_count = game_count + self.game_length_index[game_index]
        
        if game_index > len(self.game_length_index):
            print('Error: Index' + str(idx) + ' out of range for dataset of size ' + str(sum(self.game_length_index)))
        else:
            move_index = idx - sum(self.game_length_index[: game_index])
            game_file = os.listdir(self.data_dir)[game_index]
            game_file_path = self.dir_path + '/' + os.fsdecode(game_file)
            with open(game_file_path, 'rb') as f:
                game_data = np.load(f, allow_pickle=True)
                sample = game_data[move_index]
                sample['game num'] = game_index
                return(sample)

    def __len__(self):
        return(sum(self.game_length_index))