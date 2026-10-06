import pdb
import torch
import numpy as np
import torch.optim as optim


class Optimizer(object):
    def __init__(self, model, optim_dict, datasetName):
        self.optim_dict = optim_dict
        if self.optim_dict["optimizer"] == 'SGD':
            self.optimizer = optim.SGD(
                model,
                lr=self.optim_dict['base_lr'],
                momentum=0.9,
                nesterov=self.optim_dict['nesterov'],
                weight_decay=self.optim_dict['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'Adam':
            alpha = self.optim_dict['learning_ratio']
            self.optimizer = optim.Adam(
                # [
                #     {'params': model.conv2d.parameters(), 'lr': self.optim_dict['base_lr']*alpha},
                #     {'params': model.conv1d.parameters(), 'lr': self.optim_dict['base_lr']*alpha},
                #     {'params': model.rnn.parameters()},
                #     {'params': model.classifier.parameters()},
                # ],
                # model.conv1d.fc.parameters(),
                model.parameters(),
                lr=self.optim_dict['base_lr'],
                weight_decay=self.optim_dict['weight_decay']
            )
        else:
            raise ValueError()
        
        self.scheduler = self.define_lr_scheduler(self.optimizer, self.optim_dict['step'], self.optim_dict['gamma'], datasetName)

    def define_lr_scheduler(self, optimizer, milestones, gamma, datasetName):
        if self.optim_dict["optimizer"] in ['SGD', 'Adam']:

            if 'phoenix' in datasetName:
                # Warmup Scheduler
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, end_factor=1.0, total_iters=5
                )
                
                # Main Scheduler
                main_scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=milestones, gamma=0.2
                )
                
                # Switch them at epoch 5
                lr_scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer, 
                    schedulers=[warmup_scheduler, main_scheduler], 
                    milestones=[5]
                )

            elif 'Isharah' in datasetName:
                warmup_epochs = 5

                shifted_milestones = [m - warmup_epochs for m in milestones]

                # Warmup Scheduler
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
                )
                
                # Main Scheduler
                main_scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=shifted_milestones, gamma=gamma
                )
                
                # Switch them after <warmup_epochs> epochs
                lr_scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer, 
                    schedulers=[warmup_scheduler, main_scheduler], 
                    milestones=[warmup_epochs]
                )
            else:
                raise NotImplementedError(f"Dataset {datasetName} not recognized!")
            
            return lr_scheduler
        else:
            raise ValueError()    
    

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
