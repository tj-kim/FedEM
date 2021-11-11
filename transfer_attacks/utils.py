import argparse, torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from pathlib import Path

# Import General Libraries
import os
import argparse
import copy
import pickle
import random

# Import FedEM based Libraries
from utils.utils import *
from utils.constants import *
from utils.args import *
from torch.utils.tensorboard import SummaryWriter
from run_experiment import *
from models import *


class One_Hot(nn.Module):
    # from :
    # https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
    def __init__(self, depth):
        super(One_Hot,self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)
    def forward(self, X_in):
        X_in = X_in.long()
        return Variable(self.ones.index_select(0,X_in.data))
    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


def cuda(tensor,is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def rm_dir(dir_path, silent=True):
    p = Path(dir_path).resolve()
    if (not p.is_file()) and (not p.is_dir()) :
        print('It is not path for file nor directory :',p)
        return

    paths = list(p.iterdir())
    if (len(paths) == 0) and p.is_dir() :
        p.rmdir()
        if not silent : print('removed empty dir :',p)

    else :
        for path in paths :
            if path.is_file() :
                path.unlink()
                if not silent : print('removed file :',path)
            else:
                rm_dir(path)
        p.rmdir()
        if not silent : print('removed empty dir :',p)

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

def dummy_aggregator(args_):

    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_root" in args_:
        logs_root = args_.logs_root
    else:
        logs_root = os.path.join("logs", args_to_string(args_))

    print("==> Clients initialization..")
    clients = init_clients(
        args_,
        root_path=os.path.join(data_dir, "train"),
        logs_root=os.path.join(logs_root, "train")
    )

    print("==> Test Clients initialization..")
    test_clients = init_clients(
        args_,
        root_path=os.path.join(data_dir, "test"),
        logs_root=os.path.join(logs_root, "test")
    )

    logs_path = os.path.join(logs_root, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_root, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    global_learners_ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu
    )


    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]

    aggregator =\
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            lr_lambda=args_.lr_lambda,
            lr=args_.lr,
            q=args_.q,
            mu=args_.mu,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            seed=args_.seed
        )

    return aggregator, clients

# Write a Custom Class for Dataloader that has flexible batch size
class Custom_Dataloader:
    def __init__(self, x_data, y_data):
        self.x_data = x_data # Tensor + cuda
        self.y_data = y_data # Tensor + cuda
        
    def load_batch(self,batch_size,mode = 'test'):
        samples = random.sample(range(self.y_data.shape[0]),batch_size)
        out_x_data = self.x_data[samples].to(device='cuda')
        out_y_data = self.y_data[samples].to(device='cuda')
        
        return out_x_data, out_y_data
    
    def select_single(self):
        sample_idx = random.sample(range(self.y_data.shape[0]),1)
        x_point = self.x_data[sample_idx].to(device='cuda')
        y_point = self.y_data[sample_idx].to(device='cuda')
        
        return sample_idx, x_point, y_point