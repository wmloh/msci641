import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from tqdm import trange, tqdm
from sklearn.metrics import accuracy_score, f1_score
from dataset import FNCDataset, STANCE_TO_LABEL
from amodel import AModel
from pmodel import PModel
from utils.analysis import get_distribution_agree, get_distribution_polar
from variational import sample_variational
from classifier import Classifier
from inference import TestDataset
from utils.score import score_submission
from fnc_kfold import generate_features
from utils.dataset import DataSet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Experiment:
    SUBMISSION = 'submission'

    def __init__(self, unique_key, bodies_fp, stance_fp, test_body_fp, test_csv_fp, true_test_fp, resume=False):
        self.base_dir = os.path.join(self.SUBMISSION, unique_key)
        self.path_fn = lambda x: os.path.join(self.base_dir, x)
        if not resume:
            assert not os.path.isdir(self.base_dir)

        self.resume = resume

        if not os.path.isdir(self.base_dir):
            print(f'>> Creating directory -- {unique_key}')
            os.mkdir(self.base_dir)
        else:
            print(f'>> Resuming in directory -- {unique_key}')

        torch.manual_seed(0)
        np.random.seed(0)

        print('>> Initializing datasets')
        self.ds = FNCDataset(bodies_fp, stance_fp)
        self.test_ds = TestDataset(test_body_fp, test_csv_fp)
        self.true_test_label = pd.read_csv(true_test_fp)['Stance'].values
        self.amodel = None
        self.pmodel = None
        self.clf = Classifier()

        self.default_ds = DataSet(path='data')
        self.output = None

    def execute(self, params):
        try:
            start_time = time.time()

            if not self.resume or not os.path.isfile(self.path_fn('params.join')):
                with open(os.path.join(self.base_dir, 'params.json'), 'w') as f:
                    json.dump(params, f, sort_keys=False, indent=4)

            print('>> Generating agreeable and polarity dataloaders')
            params_ds_agree = params['dataset']['agree']
            # params_ds_polar = params['dataset']['polar']
            dl_agree, dl_agree_val, weights_agree = self.ds.generate_agreeable(**params_ds_agree)
            # dl_polar, dl_polar_val, weights_polar = self.ds.generate_polarity(**params_ds_polar)

            if self.resume and os.path.isfile(self.path_fn('amodel.pt')):
                print('>> Loading AModel')
                self.amodel = torch.load(self.path_fn('amodel.pt'))
            else:
                print('>> Constructing AModel')
                params_arch_amodel = params['architecture']['agree']
                self.amodel = AModel(**params_arch_amodel)

                print('>> Fitting AModel')
                params_amodel = params['model']['agree']
                amodel_loss, amodel_loss_val = AModel.fit(self.amodel, dl_agree, dl_agree_val, **params_amodel,
                                                          save_path=os.path.join(self.base_dir, 'amodel.pt'),
                                                          loss_args={
                                                              'weight': torch.from_numpy(weights_agree).to(DEVICE)})
                plt.savefig(os.path.join(self.base_dir, 'amodel_loss.png'))
                plt.close()
                self.update_output('amodel_loss', amodel_loss)
                self.update_output('amodel_loss_val', amodel_loss_val)

            df_train = pd.read_pickle('models/stance_train.pkl')
            df_val = pd.read_pickle('models/stance_val.pkl')
            X_base_train, y_base_train = generate_features(self.default_ds.stances, df_train, self.default_ds, 'train')
            X_base_val, y_base_val = generate_features(self.default_ds.stances, df_val, self.default_ds, 'val')

            X_train, y_train = AModel.combine_features(dl_agree, self.amodel, X_base_train, y_base_train)
            X_val, y_val = AModel.combine_features(dl_agree_val, self.amodel, X_base_val, y_base_val)

            if not self.resume or not os.path.isfile(self.path_fn('clf.pkl')):
                print('>> Fitting XGBoost classifier')
                self.clf.load_datasets(X_train, y_train, X_val, y_val)
                evals_result = self.clf.fit()
                self.clf.save(os.path.join(self.base_dir, 'clf.pkl'))
                self.update_output('evals_result', evals_result)
                feature_importance = self.clf.clf.get_score(importance_type='gain')
                self.update_output('feature_importance', feature_importance)
            else:
                self.clf.load(self.path_fn('clf.pkl'))

            print('>> Predicting test dataset and saving')
            X_base_test, _ = generate_features(self.default_ds.stances, None, self.default_ds,
                                               'competition', test=True)

            params_test = params['test']['pred']
            pred = self.test_ds.predict(self.amodel, self.pmodel, self.clf,
                                        X_base_test,
                                        **params_test)
            self.update_output('pred', pred)
            self.test_ds.submit(self.base_dir)

            score_test, cm_test = score_submission([STANCE_TO_LABEL[x] for x in self.true_test_label],
                                                   [STANCE_TO_LABEL[x] for x in pred])

            self.update_output('score_test', score_test)
            self.update_output('cm_test', cm_test)

            print(f'>> Test score = {score_test}')
            # print(f'>> Training score = {score} | val score = {score_val} | test score = {score_test}')

            with open(os.path.join(self.base_dir, f'score_{round(score_test)}.json'), 'w') as f:
                all_scores = {'score_test': score_test}
                json.dump(all_scores, f, sort_keys=False, indent=4)

            end_time = time.time()
            print(f'>> Execution took {round((end_time - start_time) / 60, 2)} minutes')

            with open(self.path_fn('output.pkl'), 'rb') as f:
                self.output = pickle.load(f)

            print('-' * 40)

        except ImportError as e:  # TODO: temp!
            print(e)
            return None

    def update_output(self, key, value):
        output_fp = self.path_fn('output.pkl')

        if os.path.isfile(output_fp):
            with open(output_fp, 'rb') as f:
                output = pickle.load(f)
        else:
            output = dict()

        output[key] = value
        with open(output_fp, 'wb') as f:
            pickle.dump(output, f)
