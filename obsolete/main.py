import os
import numpy as np
import torch
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

if __name__ == '__main__':
    exp = Experiment('base1', os.path.join(BASE_DIR, BODIES), os.path.join(BASE_DIR, STANCE),
                     os.path.join(BASE_DIR, TEST_BODIES), os.path.join(BASE_DIR, TEST_STANCE))
    output = exp.execute({
        'dataset': {
            'agree': {
                'batch_size': 64, 'augment_dict': {-1.: -0.3, 0.: 0.2}
            },
            'polar': {
                'batch_size': 64, 'augment_dict': {0.: 0.2}
            }
        },

        'model': {
            'agree': {
                'epochs': 1, 'display': True
            },
            'polar': {
                'epochs': 10, 'display': True
            }
        },

        'eval': True,

        'feature': {
            'train': {
                'sd_factor': 0.07, 'size': 8, 'batch_size': 16
            },
            'val': {
                'sd_factor': 0.07, 'size': 8, 'batch_size': 16
            }
        },

        'test': {
            'pred': {
                'batch_size': 64, 'sd_factor': 0.07, 'variational_size': 8,
                'v_batch_size': 16
            }
        }
    })

    s = False
    if s:
        ## TRAINING HYPERPARAMETERS
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        BATCH_SIZE = 64
        EPOCHS_A = 25
        EPOCHS_P = 90
        AMODEL_FP = os.path.join('models', 'amodel.pt')
        PMODEL_FP = os.path.join('models', 'pmodel.pt')
        FEATURE_FP = os.path.join('models', 'features_{}_{}.npy')
        CLF_FP = os.path.join('models', 'clf.pkl')
        FEATURE_TYPE = 'train'
        AMODEL_FORCE_RETRAIN = False
        PMODEL_FORCE_RETRAIN = False
        FEATURE_FORCE_GENERATE = False
        CLF_FORCE_RETRAIN = False

        ## ANALYSIS & EVALUATION
        EVAL_AMODEL = False
        EVAL_PMODEL = False

        ds = FNCDataset(os.path.join(BASE_DIR, BODIES), os.path.join(BASE_DIR, STANCE),
                        force_recompute=False)
        dl_agree, dl_agree_val = ds.generate_agreeable(batch_size=BATCH_SIZE, augment_dict={-1.: -0.3, 0.: 0.2})
        dl_polar, dl_polar_val = ds.generate_polarity(batch_size=BATCH_SIZE, augment_dict={0.: 0.2})

        if os.path.isfile(AMODEL_FP) and not AMODEL_FORCE_RETRAIN:
            amodel = torch.load(AMODEL_FP)
        else:
            amodel = AModel()
            amodel_loss, amodel_loss_val = AModel.fit(amodel, dl_agree, dl_agree_val, EPOCHS_A,
                                                      save_path=AMODEL_FP, display=True)

        if os.path.isfile(PMODEL_FP) and not PMODEL_FORCE_RETRAIN:
            pmodel = torch.load(PMODEL_FP)
        else:
            pmodel = PModel()
            pmodel_loss, pmodel_loss_val = PModel.fit(pmodel, dl_polar, dl_polar_val, EPOCHS_P,
                                                      save_path=PMODEL_FP, display=True)

        if EVAL_AMODEL:
            plt.figure()
            get_distribution_agree(amodel, dl_agree, -1., display=True)
            get_distribution_agree(amodel, dl_agree, 0., display=True)
            get_distribution_agree(amodel, dl_agree, 1., display=True)
            plt.figure()
            get_distribution_agree(amodel, dl_agree_val, -1., display=True)
            get_distribution_agree(amodel, dl_agree_val, 0., display=True)
            get_distribution_agree(amodel, dl_agree_val, 1., display=True)
        if EVAL_PMODEL:
            plt.figure()
            get_distribution_polar(pmodel, dl_polar, 0., display=True)
            get_distribution_polar(pmodel, dl_polar, 1., display=True)
            plt.figure()
            get_distribution_polar(pmodel, dl_polar_val, 0., display=True)
            get_distribution_polar(pmodel, dl_polar_val, 1., display=True)

        assert FEATURE_TYPE == 'train' or FEATURE_TYPE == 'val'
        if os.path.isfile(FEATURE_FP.format('X', FEATURE_TYPE)) and os.path.isfile(FEATURE_FP.format('y', FEATURE_TYPE)) \
                and not FEATURE_FORCE_GENERATE:
            features = np.load(FEATURE_FP.format('X', 'train'))
            labels = np.load(FEATURE_FP.format('y', 'train'))

            features_val = np.load(FEATURE_FP.format('X', 'val'))
            labels_val = np.load(FEATURE_FP.format('y', 'val'))
        else:
            if FEATURE_TYPE == 'train':
                features, labels = sample_variational(amodel, pmodel, ds.stance_train, ds.headline, ds.body, 0.07, 8,
                                                      batch_size=16, save_path=FEATURE_FP, save_path_type=FEATURE_TYPE)
            elif FEATURE_TYPE == 'val':
                features, labels = sample_variational(amodel, pmodel, ds.stance_val, ds.headline, ds.body, 0.07, 8,
                                                      batch_size=16, save_path=FEATURE_FP, save_path_type=FEATURE_TYPE)
            else:
                raise ValueError(f'{FEATURE_TYPE} is not a valid FEATURE_TYPE')

        if os.path.isfile(CLF_FP) and not CLF_FORCE_RETRAIN:
            clf = Classifier()
            clf.load(CLF_FP)
        else:
            clf = Classifier()
            clf.load_datasets(features, labels, features_val, labels_val)
            clf.fit()
            clf.save(CLF_FP)

        test_ds = TestDataset(os.path.join(BASE_DIR, TEST_BODIES), os.path.join(BASE_DIR, TEST_STANCE))
        pred = test_ds.predict(amodel, pmodel, clf, v_batch_size=16)
