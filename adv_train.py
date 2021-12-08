"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
from utils.utils import *
from utils.constants import *
from utils.args import *
from run_experiment import * 

from torch.utils.tensorboard import SummaryWriter

# Import General Libraries
import os
import argparse
import torch
import copy
import pickle
import random
import numpy as np
import pandas as pd
from models import *

# Import Transfer Attack
from transfer_attacks.Personalized_NN import *
from transfer_attacks.Params import *
from transfer_attacks.Transferer import *
from transfer_attacks.Args import *
from transfer_attacks.utils import *

from transfer_attacks.Boundary_Transferer import *
from transfer_attacks.projected_gradient_descent import *

def unnormalize_cifar10(normed):

    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.201])

    unnormalize = Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    a = unnormalize(normed)
    a = a.transpose(0,1)
    a = a.transpose(1,2)
    a = a * 255
    b = a.clone().detach().requires_grad_(True).type(torch.uint8)
    
    return b

if __name__ == "__main__":
    
    # Manually set argument parameters
    args_ = Args()
    args_.experiment = "cifar10"
    args_.method = "FedEM"
    args_.decentralized = False
    args_.sampling_rate = 1.0
    args_.input_dimension = None
    args_.output_dimension = None
    args_.n_learners= 3
    args_.n_rounds = 201
    args_.bz = 128
    args_.local_steps = 1
    args_.lr_lambda = 0
    args_.lr =0.03
    args_.lr_scheduler = 'multi_step'
    args_.log_freq = 10
    args_.device = 'cuda'
    args_.optimizer = 'sgd'
    args_.mu = 0
    args_.communication_probability = 0.1
    args_.q = 1
    args_.locally_tune_clients = False
    args_.seed = 1234
    args_.verbose = 1
    args_.save_path = 'weights/cifar/21_12_02_first_transfers_xadv_train_n40/'
    args_.validation = False

    data_save_path = 'adv_data/cifar/21_12_01_from_21_09_28_first_transfers/'
    
    # Generate the dummy values here
    aggregator, clients = dummy_aggregator(args_)
    
    # Add clients xadv dataset
    num_adv_nodes = 40 # len(aggregator.clients)
    
    for i in range(num_adv_nodes):
        dataloader_path = data_save_path + "client_" + str(i) + ".p"
        dataloader = pickle.load( open(dataloader_path, "rb" ) )
        
        # Convert image to correct format one by one 
        num_img = dataloader.x_data.shape[0]
        data_xn = []
        
        for j in range(num_img):
            img = dataloader.x_data[j]
            x_new = unnormalize_cifar10(img)
            
            data_xn.append(x_new)
        
        data_xn = torch.stack(data_xn)
        data_yn = dataloader.y_data
        
        aggregator.clients[i].train_iterator.dataset.data = data_xn
        aggregator.clients[i].train_iterator.dataset.targets = data_yn

    # Train the model
    print("Training..")
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0
    while current_round <= args_.n_rounds:

        aggregator.mix()

        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round

    if "save_path" in args_:
        save_root = os.path.join(args_.save_path)

        os.makedirs(save_root, exist_ok=True)
        aggregator.save_state(save_root)
            