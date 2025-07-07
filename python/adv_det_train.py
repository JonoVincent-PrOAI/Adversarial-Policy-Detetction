
import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from adv_det_dataset import Adversarial_Detection_Dataset as adv_dataset
from adv_det_model import Model

class Training_Loop():

    def __init__(self, data_dir_path : str, batch_size : int, num_workers : int):

        self.batch_size = batch_size

        self.dataset = adv_dataset(data_dir_path)

        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])

        meta_dir = os.path.dirname(data_dir_path)

        meta_data_path =  meta_dir +'/'+'meta_data.json'
        with open(meta_data_path, 'r') as f:
            meta_data = json.load(f)
            f.close()
            kata_model_config = meta_data['model data']

        self.model = Model(kata_model_config = kata_model_config, pos_len=19)

        self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers = num_workers)

        self.val_loader = DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle=True, num_workers = num_workers)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self. optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)


    def train_epoch(self, epoch_index : int, tb_writer:SummaryWriter):

        running_loss = 0.
        last_loss = 0

        training_loss = []
        eval_loss = []
        eval_accuracy = []

        for i, data in enumerate(self.train_loader):

            inputs, labels = data

            labels = torch.tensor(labels)

            labels = labels.long()

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            outputs = outputs.float()
            
            loss = self.loss_fn(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            last_loss = running_loss / self.batch_size # average loss per batch

            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(self.train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            training_loss.append(last_loss)
            running_loss = 0.

        eval_epoch_loss, eval_epoch_accuracy = self.eval_loop()
        eval_loss.append(eval_epoch_loss)
        eval_accuracy.append(eval_epoch_accuracy)

        return (last_loss,training_loss,eval_loss, eval_accuracy)
    
    def eval_loop(self):

        self.model.eval()

        correct = 0

        last_loss = 0

        running_loss = 0

        loss_fn = torch.nn.CrossEntropyLoss()

        with torch.no_grad():

            for i, data in enumerate(self.val_loader):

                inputs, labels = data

                outputs = self.model(inputs)

                labels = torch.tensor(labels)

                labels = labels.long()

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                outputs = outputs.float()

                loss = loss_fn(outputs, labels)

                predicted = [0 if n[0] > n[1] else 1 for n in outputs]

                correct = [1 if p == l else 0 for p,l in zip(predicted, labels)]

                correct = sum(correct)

                running_loss += loss.item()
        
        eval_loss = running_loss/len(labels)
        accuracy = correct/len(labels)

        return(eval_loss, accuracy)




    


