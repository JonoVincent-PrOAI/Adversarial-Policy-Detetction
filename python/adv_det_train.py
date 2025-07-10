
import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from adv_det_dataset import Adversarial_Detection_Dataset as adv_dataset
from adv_det_model import Model
import wandb

class Training_Loop():

    def __init__(self, data_dir_path : str, batch_size : int, num_workers : int, learing_rate):

        self.batch_sz = batch_size

        self.lr = learing_rate

        self.dataset = adv_dataset(data_dir_path)

        train_size = int(0.8 * len(self.dataset))
        self.val_size = len(self.dataset) - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, self.val_size])

        meta_dir = os.path.dirname(data_dir_path)

        meta_data_path =  meta_dir +'/'+'meta_data.json'
        with open(meta_data_path, 'r') as f:
            meta_data = json.load(f)
            f.close()
            kata_model_config = meta_data['model data']

        self.model = Model(kata_model_config = kata_model_config, pos_len=19)

        self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_sz, shuffle=True, num_workers = num_workers)

        self.val_loader = DataLoader(self.val_dataset, batch_size = self.batch_sz, shuffle=True, num_workers = num_workers)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self. optimizer = torch.optim.Adam(self.model.parameters(), lr = learing_rate)


    def train(self, num_epochs : int):

        for i in range(num_epochs):

            self.train_epoch(i)
            torch.save(self.model, 'python/Adv Pol Det Models')



    def train_and_log(self, num_epochs : int, run_name : str, wandb_key : str):

        wandb.login(key = wandb_key)

        run = wandb.init(
            project = "Adversarial_Detection_Test",
            config={
                "learning rate" : self.lr,
                "epochs" : num_epochs,
                "batch size" : self.batch_sz,
            },
        )

        for i in range(num_epochs):

            last_loss, training_loss, eval_loss, eval_accuracy = self.train_epoch(i)
            wandb.log({'training loss': last_loss, 'eval accuracy' : eval_accuracy, 'eval loss' : eval_loss})

            save_path = 'Adv_Pol_Det_Models/' + '/model_' + str(i) + '.chkpt'
            torch.save(self.model, save_path)
        
        wandb.finish()



    def train_epoch(self, epoch_index : int):

        running_loss = 0.
        average_batch_loss = 0.

        training_loss = []
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

            average_batch_loss = running_loss / self.batch_sz

            #print('  batch {} loss: {}'.format(i + 1, average_batch_loss))
            training_loss.append(average_batch_loss)
            running_loss = 0.
        
        last_batch_loss = average_batch_loss
        eval_loss, eval_accuracy = self.eval_loop()
        print("Acc: " + str(eval_accuracy))

        return (last_batch_loss,training_loss,eval_loss, eval_accuracy)
    
    def eval_loop(self):

        self.model.eval()

        correct = []

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

                correct.append([1 if p == l else 0 for p,l in zip(predicted, labels)])

                running_loss += loss.item()
        
        eval_loss = running_loss/self.val_size

        num_correct = sum([sum(n) for n in correct])

        accuracy = float(num_correct)/float(self.val_size)

        return(eval_loss, accuracy)




    


