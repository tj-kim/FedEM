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
    

    exp_names = ['FedAvg', 'FedAvg', 'FedAvg']
    exp_savenames = ['Uw', 'URsv', 'UGoT']
    exp_name = '23_06_23_DistMEC_FL/'
    n_vals = 1

    # Load the pickled file and check for participation
    with open("/home/ubuntu/FedEM/23_06_22_participant_array.pkl", "rb") as tf:
        loaded_dict = pickle.load(tf)

    # Access the loaded dictionary
    participant_list = []
    participant_list += [loaded_dict['Users_w_sa']]
    participant_list += [loaded_dict['Users_rsv_sa']]
    participant_list += [loaded_dict['GoT_Users_sa']]


    # Manually set argument parameters
    args_ = Args()
    args_.experiment = "cifar10"
    args_.method = "FedAvg"
    args_.decentralized = False
    args_.sampling_rate = 1.0
    args_.input_dimension = None
    args_.output_dimension = None
    args_.n_learners= n_vals
    args_.n_rounds = 999 # Reduced number of steps
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

    # Other Argument Parameters
    reward_threshold = 0.4
    num_clients = 16

    for itt in range(len(exp_names)):
        print("running trial:", itt, "out of", len(exp_names)-1)

        # Calculate 

        args_.save_path = 'weights/DisMEC/' + exp_name + exp_savenames[itt]

        # Generate the dummy values here
        aggregator, clients = dummy_aggregator_distmec(args_, num_clients)


        # Train the model
        print("Training..")
        pbar = tqdm(total=args_.n_rounds)
        current_round = 0
        while current_round <= args_.n_rounds:

            # Extract participant id # list 
            participant_sub = participant_list[itt][:,current_round]
            participant_id = np.where(participant_sub > 0.4)[0]

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

        # Pickle aggregator
        train_log_save_path = args_.save_path + '/train_log.p'
        aggregator.global_train_logger.close()

        with open(train_log_save_path, 'wb') as handle:
            pickle.dump(aggregator.acc_log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


        del aggregator, clients
        torch.cuda.empty_cache()
            