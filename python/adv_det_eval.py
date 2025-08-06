import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from adv_det_dataset import Adversarial_Detection_Dataset as adv_dataset
from adv_det_model import Model
import wandb
import random

class Evaluation():

    def __init__(self, model, data_path, batch_sz, num_workers):

        self.batch_sz = batch_sz

        self.dataset = adv_dataset(data_path)
        self.dataloader = DataLoader(self.dataset, batch_size = self.batch_sz, shuffle=False, num_workers = num_workers)

        self.model = model

    

    def evaluate_model(self):

        self.model.eval()

        correct_per_move_num = {}
        correct_per_move_adv = {}
        correct_per_move_non_adv = {}
        correct_per_adv_model = {}

        adv_model_names = ['atari', 'base-adv', 'cont', 'gift', 'large', 'stall', 'ViT-adv']
        for model in adv_model_names:
            correct_per_adv_model[model] = []


        for i, (indexes, inputs, labels) in enumerate(self.dataloader):

            for j, (idx, input, label) in enumerate(zip(indexes, inputs, labels)):

                label = label.long()

                output = self.model(input)[0]
                output = output.float()

                if output[0] > output[1]:
                    predicted = 0.0
                else: 
                    predicted = 1.0

                correct = int(predicted == label)

                move_num = self.dataset.get_sample(idx)['move num']

                if (move_num) in correct_per_move_num.keys():
                    correct_per_move_num[(move_num)].append(correct)
                else:
                    correct_per_move_num[(move_num)] = [correct]

                if label == 0.0:
                    if (move_num) in correct_per_move_non_adv.keys():
                        correct_per_move_non_adv[(move_num)].append(correct)
                    else:
                        correct_per_move_non_adv[(move_num)] = [correct]
                else:
                    if (move_num) in correct_per_move_adv.keys():
                        correct_per_move_adv[(move_num)].append(correct)
                    else:
                        correct_per_move_adv[(move_num)] = [correct]
                        
                    
                    meta_data = self.dataset.get_meta_data_from_idx(idx)

                    model1 = meta_data[0]
                    model2 = meta_data[1]
                    adv = meta_data[-1]
                    if model1 in adv_model_names:
                        correct_per_adv_model[model1].append(correct)
                    elif model2 in adv_model_names:
                        correct_per_adv_model[model2].append(correct)
                    else:
                        if model1 != 'NA':
                            print('error uknow model :' + model1)
                        if model2 != 'NA':
                            print('error uknow model :' + model1)                

        return(correct_per_move_num, correct_per_move_adv, correct_per_move_non_adv, correct_per_adv_model)
