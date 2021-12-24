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
    args_.n_rounds = 10
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
    args_.save_path = 'weights/cifar/21_09_28_first_transfers/'
    args_.validation = False

    data_save_path = 'adv_data/cifar/21_12_01_from_21_09_28_first_transfers/'
    
    # Generate the dummy values here
    aggregator, clients = dummy_aggregator(args_)
    # Import weights for aggregator
    aggregator.load_state(args_.save_path)
    
    # Generate attacker --> load weights, load vals, attack
    # This is where the models are stored -- one for each mixture --> learner.model for nn
    hypotheses = aggregator.global_learners_ensemble.learners

    # obtain the state dict for each of the weights 
    weights_h = []

    for h in hypotheses:
        weights_h += [h.model.state_dict()]
    
    # Set model weights
    weights = np.load(args_.save_path + "/train_client_weights.npy")
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    model_weights = []
    num_models = len(clients)

    for i in range(num_models):
        model_weights += [weights[i]]

    # Generate the weights to test on as linear combinations of the model_weights
    models_test = []

    for (w0,w1,w2) in model_weights:
        # first make the model with empty weights
        new_model = copy.deepcopy(hypotheses[0].model)
        new_model.eval()
        new_weight_dict = copy.deepcopy(weights_h[0])
        for key in weights_h[0]:
            new_weight_dict[key] = w0*weights_h[0][key] + w1*weights_h[1][key] + w2*weights_h[2][key]
        new_model.load_state_dict(new_weight_dict)
        models_test += [new_model]
        
    # Save trainerloader to each of the pkl file based on client
    for i in range(len(models_test)):
        print("attacking client",i)
        #xdata = aggregator.clients[i].train_iterator.dataset.data
        # ydata = aggregator.clients[i].train_iterator.dataset.targets
        
    
        data_x = []
        data_y = []

        daniloader = clients[i].train_iterator
        for (x,y,idx) in daniloader.dataset:
            data_x.append(x)
            data_y.append(y)

        xdata = torch.stack(data_x)
        ydata = torch.stack(data_y)
        cdloader = Custom_Dataloader(xdata,ydata)
    
        # Attack parameters
        atk_params = PGD_Params()
        atk_params.set_params(batch_size=500, iteration = 30,
                       target = 9, x_val_min = torch.min(xdata), x_val_max = torch.max(xdata),
                       step_size = 0.05, step_norm = "inf", eps = 4.5, eps_norm = 2)

        now_network = Adv_NN(models_test[i], cdloader)
        now_network.pgd_sub(atk_params=atk_params, x_in=xdata, y_in=ydata, x_base = None)

        
        xadv_temp = now_network.x_adv
        
        now_network.forward_transfer(x_orig = xdata.cuda(), x_adv  = xadv_temp.cuda(), y_orig = ydata.cuda(), y_adv = ydata.cuda(),
                         true_labels = ydata.cuda(), target = atk_params.target, print_info = False)
        
        print("ADV Hit Rate:", now_network.adv_target_achieve)
        # Pickle Save
        name_cdloader = data_save_path + "client_" + str(i) + ".p"
        new_cdloader = Custom_Dataloader(xadv_temp, ydata)
        pickle.dump( new_cdloader, open( name_cdloader, "wb" ) )
        