import os
import random
from collections import namedtuple
import numpy as np
import time
from datetime import datetime

import torch

from datasets.feddata import FedData

from algorithms.fedavg import FedAvg
from algorithms.fedavgCL import FedAvgCL
from algorithms.fedreg import FedReg
from algorithms.scaffold import Scaffold

from algorithms.fedopt import FedOpt
from algorithms.fednova import FedNova
from algorithms.moon import MOON
from algorithms.feddyn import FedDyn

from algorithms.pfedme import pFedMe
from algorithms.perfedavg import PerFedAvg

from algorithms.fedphp import FedPHP

from algorithms.flce import FLCE
from algorithms.fedfa import FedFa
from algorithms.fedfv import FedFv
from algorithms.fedsharplyavg import FedSharplyAvg
from algorithms.qfedavg import qFedAvg
from algorithms.ditto import Ditto
from algorithms.fedproavg import FedproAvg
from algorithms.CGSV import CGSV
from algorithms.fedmdfg import FedMDFG


from networks.basic_nets import VGG, ResNet, ResNet18, TFCNN, ModifiedResNet18, CNN_CIFAR10_FedAvg

from paths import save_dir
from config import default_param_dicts

from utils import weights_init
from utils import setup_seed

torch.set_default_tensor_type(torch.FloatTensor)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def construct_model(args):
    # model = ModifiedResNet18(num_classes=10)
    model = ResNet(
        n_layer=args.n_layer,
        n_classes=args.n_classes,
        # use_bn=False
    )
    # model = TFCNN(n_classes=args.n_classes)
    # model = CNN_CIFAR10_FedAvg(n_classes=args.n_classes)

    model.apply(weights_init)
    return model


def construct_algo(args):
    if args.algo == "fedavg":
        FedAlgo = FedAvg
    elif args.algo == "fedavgcl":
        FedAlgo = FedAvgCL
    elif args.algo == "ditto":
        FedAlgo = Ditto
    elif args.algo == "qfedavg":
        FedAlgo = qFedAvg
    elif args.algo == "fedprox":
        FedAlgo = FedReg
    elif args.algo == "fedmdfg":
        FedAlgo = FedMDFG
    elif args.algo == "fedproavg":
        FedAlgo = FedproAvg
    elif args.algo == "fedmmd":
        FedAlgo = FedReg
    elif args.algo == "scaffold":
        FedAlgo = Scaffold
    elif args.algo == "fedopt":
        FedAlgo = FedOpt
    elif args.algo == "fednova":
        FedAlgo = FedNova
    elif args.algo == "moon":
        FedAlgo = MOON
    elif args.algo == "feddyn":
        FedAlgo = FedDyn
    elif args.algo == "pfedme":
        FedAlgo = pFedMe
    elif args.algo == "perfedavg":
        FedAlgo = PerFedAvg
    elif args.algo == "fedphp":
        FedAlgo = FedPHP
    elif args.algo == "flce":
        FedAlgo = FLCE
    elif args.algo == "fedfa":
        FedAlgo = FedFa
    elif args.algo == "fedfv":
        FedAlgo = FedFv
    elif args.algo == "CGSV":
        FedAlgo = CGSV
    elif args.algo == "fedsharplyavg":
        FedAlgo = FedSharplyAvg
    else:
        raise ValueError("No such fed algo:{}".format(args.algo))
    return FedAlgo


def get_hypers(algo):
    if algo == "fedavg":
        hypers = {
            "cnt": 1,      #循环轮次
            "none": ["none"] * 2
        }
    elif algo == "fedavgcl":
        hypers = {
            "cnt": 1,      #循环轮次
            "none": ["none"] * 2
        }
    elif algo == "ditto":
        hypers = {
            "cnt": 1,      #循环轮次
            "none": ["none"] * 2
        }
    elif algo == "CGSV":
        hypers = {
            "cnt": 1,      #循环轮次
            "none": ["none"] * 2
        }
    elif algo == "qfedavg":
        hypers = {
            "cnt": 1,      #循环轮次
            "none": ["none"] * 2
        }
    elif algo == "fedprox":
        hypers = {
            "cnt": 5,
            "reg_way": ["fedprox"] * 5,
            "reg_lamb": [1e-5, 1e-1, 1e-4, 1e-3, 1e-2]
        }
    elif algo == "fedmmd":
        hypers = {
            "cnt": 4,
            "reg_way": ["fedmmd"] * 4,
            "reg_lamb": [1e-2, 1e-3, 1e-4, 1e-1]
        }
    elif algo == "fedmdfg":
        hypers = {
            "cnt": 4,
            "reg_way": ["fedmmd"] * 4,
            "reg_lamb": [1e-2, 1e-3, 1e-4, 1e-1]
        }
    elif algo == "scaffold":
        hypers = {
            "cnt": 2,
            "glo_lr": [0.25, 0.5]
        }
    elif algo == "fedopt":
        hypers = {
            "cnt": 8,
            "glo_optimizer": [
                "SGD", "Adam", "SGD", "SGD", "Adam", "SGD", "SGD", "Adam"
            ],
            "glo_lr": [0.1, 3e-4, 0.05, 0.01, 1e-4, 0.3, 0.03, 5e-5],
        }
    elif algo == "fednova":
        hypers = {
            "cnt": 8,
            "gmf": [0.5, 0.1, 0.5, 0.5, 0.1, 0.5, 0.75, 0.9],
            "prox_mu": [1e-3, 1e-3, 1e-4, 1e-2, 1e-4, 1e-5, 1e-4, 1e-3],
        }
    elif algo == "moon":
        hypers = {
            "cnt": 8,
            "reg_lamb": [1e-4, 1e-2, 1e-3, 1e-1, 1e-5, 1.0, 5e-4, 5e-3]
        }
    elif algo == "feddyn":
        hypers = {
            "cnt": 8,
            "reg_lamb": [1e-3, 1e-2, 1e-4, 1e-1, 1e-5, 1e-7, 1e-6, 5e-5]
        }
    elif algo == "pfedme":
        hypers = {
            "cnt": 8,
            "reg_lamb": [1e-4, 1e-2, 1e-3, 1e-5, 1e-4, 1e-5, 1e-5, 1e-4],
            "alpha": [0.1, 0.75, 0.5, 0.25, 0.5, 1.0, 0.75, 0.9],
            "k_step": [20, 10, 20, 20, 10, 5, 5, 10],
            "beta": [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0],
        }
    elif algo == "perfedavg":
        hypers = {
            "cnt": 5,
            "meta_lr": [0.05, 0.01, 0.1, 0.03, 0.005],
        }
    elif algo == "fedphp":
        hypers = {
            "cnt": 3,
            "reg_way": ["KD", "MMD", "MMD"],
            "reg_lamb": [0.1, 0.1, 0.05],
        }
    elif algo == "flce":
        hypers = {
            "cnt": 1,
            "none": ["none"] * 2
        }
    elif algo == "fedproavg":
        hypers = {
            "cnt": 1,
            "none": ["none"] * 2
        }
    elif algo == "fedfa":
        hypers = {
            "cnt": 1,
            "none": ["none"] * 2
        }
    elif algo == "fedfv":
        hypers = {
            "cnt": 1,
            "none": ["none"] * 2
        }
    elif algo == "fedsharplyavg":
        hypers = {
            "cnt": 1,      #循环轮次
            "none": ["none"] * 2
        }
    else:
        raise ValueError("No such fed algo:{}".format(algo))
    return hypers


def main_federated(para_dict):

    print(para_dict)

    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    # DataSets
    try:
        n_clients = args.n_clients
    except Exception:
        n_clients = None

    try:
        nc_per_client = args.nc_per_client
    except Exception:
        nc_per_client = None

    try:
        dir_alpha = args.dir_alpha
    except Exception:
        dir_alpha = None

    feddata = FedData(
        dataset=args.dataset,
        split=args.split,
        n_clients=n_clients,
        nc_per_client=nc_per_client,
        dir_alpha=dir_alpha,
        n_max_sam=args.n_max_sam,
        filename=args.filename
    )
    csets, gset = feddata.construct()   #

    list = []
    for i in csets.values():
        client_data = len

    try:
        nc = int(args.dset_ratio * len(csets))
        clients = list(csets.keys())
        sam_clients = np.random.choice(
            clients, nc, replace=False
        )
        csets = {
            c: info for c, info in csets.items() if c in sam_clients
        }

        n_test = int(args.dset_ratio * len(gset.xs))
        inds = np.random.permutation(len(gset.xs))
        gset.xs = gset.xs[inds[0:n_test]]
        gset.ys = gset.ys[inds[0:n_test]]

    except Exception:
        pass

    feddata.print_info(csets, gset)

    # Model
    model = construct_model(args)
    print(model)
    print([name for name, _ in model.named_parameters()])
    n_params = sum([
        param.numel() for param in model.parameters()
    ])
    print("Total number of parameters : {}".format(n_params))

    if args.cuda:
        model = model.cuda()

    FedAlgo = construct_algo(args)
    algo = FedAlgo(
        csets=csets,
        gset=gset,
        model=model,
        args=args
    )

    algo.train()


    fpath = os.path.join(
        save_dir, args.fname
    )
    print('--------------------save location:  {}--------------------'.format(fpath))
    algo.save_logs(fpath)
    print(algo.logs)


def main_cifar_dir(dataset, algo):
    hypers = get_hypers(algo)      #根据不同算法获取超参

    lr = 0.01                      #学习率
    # for filename in ['Cifar10_IID_numc50_alpha0_seed0_gaussian0_randomother0.2_dualityFalse_fraction','Cifar10_IID_numc50_alpha0_seed0_gaussian0.2_randomother0_dualityFalse_fraction','Cifar10_IID_numc50_alpha0_seed0_gaussian0.2_randomother0.2_dualityTrue_fraction',]:
    # for filename in ['Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0_randomother0.2_dualityFalse_fraction','Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.2_randomother0_dualityFalse_fraction','Cifar10_NonIID_numc50_alpha0.5_seed0_gaussian0.2_randomother0.2_dualityTrue_fraction',]:
    # for filename in ['EuroSAT_NonIID_numc50_alpha0.1_seed0_unbalanced', 'EuroSAT_NonIID_numc50_alpha0.3_seed0_unbalanced','EuroSAT_NonIID_numc50_alpha0.8_seed0_unbalanced','EuroSAT_NonIID_numc50_alpha1_seed0_unbalanced','EuroSAT_NonIID_numc50_alpha10_seed0_unbalanced','EuroSAT_NonIID_numc50_alpha100_seed0_unbalanced',]:
    # for filename in ['Cifar10_IID_numc50_alpha0_seed0_unbalanced_gaussian0_randomother0.2_dualityFalse_fraction','Cifar10_IID_numc50_alpha0_seed0_unbalanced_gaussian0.2_randomother0_dualityFalse_fraction','Cifar10_IID_numc50_alpha0_seed0_unbalanced_gaussian0.2_randomother0.2_dualityTrue_fraction']:
    for filename in ['Cifar100_IID_numc50_alpha0_seed0', 'Cifar100_NonIID_numc50_alpha0.3_seed0', 'Cifar100_NonIID_numc50_alpha0.8_seed0', 'Cifar100_NonIID_numc50_alpha1_seed0', 'Cifar100_NonIID_numc50_alpha10_seed0', 'Cifar100_NonIID_numc50_alpha100_seed0', ]:
    # for filename in ['Cifar10_NonIID_numc50_alpha0.1_seed0', 'Cifar10_NonIID_numc50_alpha0.3_seed0','Cifar10_NonIID_numc50_alpha0.8_seed0', 'Cifar10_NonIID_numc50_alpha1_seed0', 'Cifar10_NonIID_numc50_alpha10_seed0', 'Cifar10_NonIID_numc50_alpha100_seed0']:
    # for filename in ['EuroSAT_NonIID_numc50_alpha0.1_seed0', 'EuroSAT_NonIID_numc50_alpha0.3_seed0', 'EuroSAT_NonIID_numc50_alpha0.5_seed0', 'EuroSAT_NonIID_numc50_alpha0.8_seed0', 'EuroSAT_NonIID_numc50_alpha1_seed0', 'EuroSAT_NonIID_numc50_alpha10_seed0', 'EuroSAT_NonIID_numc50_alpha100_seed0', 'EuroSAT_IID_numc50_alpha0_seed0']:
        for dir_alpha in [100, 10]:
            for j in range(hypers["cnt"]):
                para_dict = {}
                for k, vs in default_param_dicts[dataset].items():
                    para_dict[k] = random.choice(vs)

                para_dict["algo"] = algo                #算法
                para_dict["dataset"] = dataset          #数据集
                para_dict["n_layer"] = 20                #网络层数
                para_dict["split"] = "dirichlet"        #数据集切分方法
                para_dict["dir_alpha"] = dir_alpha      #迪利克雷分布系数
                para_dict["lr"] = lr                    #学习率
                # para_dict['max_grad_norm'] = 100        #最大梯度（防止范数过大）
                # para_dict['weight_decay'] = 1e-5        #参数正则化（防止范数过大）
                para_dict["n_clients"] = 50            #客户端数量
                para_dict["c_ratio"] = 0.2              #客户端比例
                # para_dict["local_epochs"] = 5           #本地训练轮次（适用于每个客户端的数据量不同的情况）
                para_dict["local_steps"] = 50           #本地训练轮次（适用于每个客户端的数据量相同的情况）
                para_dict["max_round"] = 1000           #最大轮次
                para_dict["test_round"] = 1            #测试轮次
                para_dict["filename"] = filename

                for key, values in hypers.items():
                    if key == "cnt":
                        continue
                    else:
                        para_dict[key] = values[j]

                para_dict["fname"] = "{}-K100-Dir-{}-ResNet20-{}-1000.log".format(
                    para_dict["filename"], 'mydata', 'FLCE'
                    # dataset, dir_alpha, algo + '_CE_CL+V'
                )

        main_federated(para_dict)


if __name__ == "__main__":
    # set seed
    setup_seed(seed=0)

    # algos = [
    #     "fedproavg", "fedavg", "fedprox", "fedmmd", "scaffold",
    #     "fedopt", "fednova", "moon", "feddyn",
    #     "perfedavg", "pfedme", "fedphp", "fedfa", "fedfv", "fedsharplyavg"
    # ]
    # dataset = [
    #     "cifar10_mydata", "cifar10_Allmydata", "cifar10_unbalanced_mydata", "cifar10_unbalanced_Allmydata",
    #     "Eurosat_mydata", "Eurosat_Allmydata", "Eurosat_unbalanced_mydata", "Eurosat_unbalanced_Allmydata",
    #     "cifar100_mydata", "cifar100_Allmydata", "cifar100_unbalanced_mydata", "cifar100_unbalanced_Allmydata",
    # ]

    dataset = ""
    algos = ["flce", ]
    # main_cifar_dir(dataset, algo)

    #设定数据集和算法
    for dataset in ['cifar100_Allmydata']:
        for algo in algos:
            main_cifar_dir(dataset, algo)
