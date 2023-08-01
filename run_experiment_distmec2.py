"""Run Experiment
FL overlay on Distmec - clients below threshold drop out of training
6/22/23 
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
from transfer_attacks.TA_utils import *

import numba 


if __name__ == "__main__":
    # Define the list of pickled dictionary file paths
    participant_files = [
        "/home/ubuntu/FedEM/distmec_participant_pkls/femnist/23_07_30_participant_array_femnist_50c_hard0.pkl",
        "/home/ubuntu/FedEM/distmec_participant_pkls/femnist/23_07_30_participant_array_femnist_50c_hard1.pkl",
        "/home/ubuntu/FedEM/distmec_participant_pkls/femnist/23_07_30_participant_array_femnist_50c_hard2.pkl",
        "/home/ubuntu/FedEM/distmec_participant_pkls/femnist/23_07_30_participant_array_femnist_50c_hard3.pkl",
        "/home/ubuntu/FedEM/distmec_participant_pkls/femnist/23_07_30_participant_array_femnist_50c_hard4.pkl"
    ]
    

    exp_names = ['FedAvg', 'FedAvg', 'FedAvg']
    exp_savenames = ['Uw', 'URsv', 'UGoT']
    exp_name = '23_07_30_DistMEC_SL_newmodel_50c_femnist_300t_data10_hard/'
    n_vals = 1
    
    num_clients = 50
    offset_expr = num_clients
    rounds_max = 300 + num_clients # based on loaded trace
    client_data_proportion = 1
    
    # Manually set argument parameters
    args_ = Args()
    args_.experiment = "femnist"
    args_.method = "FedAvg"
    args_.decentralized = False
    args_.sampling_rate = 1.0
    args_.input_dimension = None
    args_.output_dimension = None
    args_.n_learners= n_vals
    args_.n_rounds = rounds_max-offset_expr # Reduced number of steps
    args_.bz = 128
    args_.local_steps = 1
    args_.lr_lambda = 0
    args_.lr =0.01
    args_.lr_scheduler = 'multi_step'
    args_.log_freq = 5
    args_.device = 'cuda'
    args_.optimizer = 'sgd'
    args_.mu = 0
    args_.communication_probability = 0.1
    args_.q = 1
    args_.locally_tune_clients = False
    args_.seed = 1234
    args_.verbose = 1
    args_.validation = False
    args_.save_freq = 3
    
    reward_threshold = 0.0

    for itt, participant_file in itertools.product(range(len(exp_names)), participant_files):
        print("Running iteration:", "(", itt+1, "," , int(participant_file[-5]) + 1, ") ", "out of", "(", len(exp_names), "," , len(participant_files), ") ")

        # Load the pickled file and check for participation
        with open(participant_file, "rb") as tf:
            loaded_dict = pickle.load(tf)

        # Access the loaded dictionary
        participant_list = []
        participant_list += [loaded_dict['Users_w_sa']]
        participant_list += [loaded_dict['Users_rsv_sa']]
        participant_list += [loaded_dict['GoT_Users_sa']]

        # Rest of the code remains the same

        # Calculate 

        args_.save_path = 'weights/DisMEC/' + exp_name + exp_savenames[itt]

        # Generate the dummy values here
        aggregator, clients = dummy_aggregator_distmec(args_, num_clients)

        # Alter client data INSIDE AGGREGATOR to be proportoinally less
#         update_aggregator_dataset(aggregator, client_data_proportion)

        # Rest of the code remains the same        
        # Train the model
        print("Training..")
        pbar = tqdm(total=args_.n_rounds)
        current_round = 0
        while current_round <= args_.n_rounds:

            # Extract participant id # list 
            participant_sub = participant_list[itt][:,current_round+offset_expr]
            participant_id = np.where(participant_sub > reward_threshold)[0]

            if len(participant_id) == 0:
                aggregator.c_round += 1
                current_round = aggregator.c_round
                if aggregator.c_round % aggregator.log_freq == 0:
                    aggregator.write_logs()
            else:
                aggregator.mix_partial(participant_id)            

                if aggregator.c_round != current_round:
                    pbar.update(1)
                    current_round = aggregator.c_round

        if "save_path" in args_:
            save_root = os.path.join(args_.save_path)

            os.makedirs(save_root, exist_ok=True)
            aggregator.save_state(save_root)
            
        # Save the train log
        train_log_save_path = args_.save_path + '/train_log' + participant_file[-5] + '.p'
        aggregator.global_train_logger.close()

        with open(train_log_save_path, 'wb') as handle:
            pickle.dump(aggregator.acc_log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        del aggregator, clients
        torch.cuda.empty_cache()
