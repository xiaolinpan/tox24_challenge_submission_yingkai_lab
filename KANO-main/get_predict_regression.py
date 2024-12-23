import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from argparse import Namespace
import csv
from logging import Logger
import os
from typing import List
import pandas as pd

import numpy as np
from tensorboardX import SummaryWriter
import torch
import pickle
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from chemprop.train.evaluate import evaluate, evaluate_predictions
from chemprop.train.predict import predict
from chemprop.train.train import train
from chemprop.data import StandardScaler
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import build_model, build_pretrain_model, add_functional_prompt
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, makedirs, save_checkpoint
from chemprop.data import MoleculeDataset
from tqdm import tqdm, trange
from chemprop.models import ContrastiveLoss
from chemprop.torchlight import initialize_exp, snapshot
from chemprop.utils import load_args, load_checkpoint, load_scalers
from torch.optim import Adam
from chemprop.parsing import parse_predict_args
from torch.optim.lr_scheduler import ExponentialLR
from chemprop.train.run_training import run_training
from itertools import chain

if __name__ == '__main__':
    args = parse_predict_args()
    embs = []
    #train_data = get_data(path=args.data_path, args=args)
    #train_smiles, train_targets = train_data.smiles(), train_data.targets()
    #train_scaler = StandardScaler().fit(train_targets)
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
    # Load model
        model = load_checkpoint(checkpoint_path, cuda=args.cuda)
        input_data = get_data(path=args.test_path, args=args)
        test_preds = predict(
                        model=model,
                        prompt=False,
                        data=input_data,
                        batch_size=256,
                        #scaler=train_scaler
                        scaler=scaler # add
                        )
        predict_proba = list(chain.from_iterable(test_preds))
        embs.append(predict_proba)
    
    #print(embs[:5])
    test_file=args.test_path.split('/')[-1].split('.')[0]
    transposed_data = zip(*embs)
    embs_preds = [sum(column) / len(column) for column in transposed_data]
    #print(embs_preds[:5])
    df_out=pd.DataFrame()
    #df_out['smiles']=smiles
    df_out['KANO_preds'] = embs_preds
    df_out['KANO_preds']=df_out['KANO_preds'].astype(float)
    save_path=args.preds_path
    df_out.to_csv(os.path.join(save_path, '{}_KANO_prediction.csv'.format(test_file)), index=False)