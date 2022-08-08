import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import trange, tqdm
from sklearn.metrics import accuracy_score, f1_score
from dataset import FNCDataset
from amodel import AModel
from pmodel import PModel
from utils.analysis import get_distribution_agree, get_distribution_polar
from variational import sample_variational
from classifier import Classifier
from inference import TestDataset
from experiment import Experiment
from utils.score import score_submission

## SETUP CONSTANTS
BASE_DIR = 'data'
STANCE = 'train_stances.csv'
BODIES = 'train_bodies.csv'
TEST_STANCE = 'competition_test_stances_unlabeled.csv'
TEST_BODIES = 'competition_test_bodies.csv'
TRUE_TEST = 'competition_test_stances.csv'

if __name__ == '__main__':
    exp = Experiment('combine_simblock2', os.path.join(BASE_DIR, BODIES), os.path.join(BASE_DIR, STANCE),
                     os.path.join(BASE_DIR, TEST_BODIES), os.path.join(BASE_DIR, TEST_STANCE),
                     os.path.join(BASE_DIR, TRUE_TEST), resume=True)
    exp.execute({
        'dataset': {
            'agree': {
                'batch_size': 64, 'augment_dict': {-1.: -0.4, 0.: 0.2, 1.: 0.05}
            },
        },
        'architecture': {
            'agree': {
                'dropout': 0.1,
                'num_simblocks': 2,
                'embedding_size': 16,
            },
        },
        'model': {
            'agree': {
                'epochs': 25, 'display': True
            },
        },
        'test': {
            'pred': {
                'batch_size': 64,
            }
        }
    })
    # del exp
    # torch.cuda.empty_cache()
