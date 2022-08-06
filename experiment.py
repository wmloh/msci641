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

    def execute(self, params):
        try:
            start_time = time.time()

            if not self.resume or not os.path.isfile(self.path_fn('params.join')):
                with open(os.path.join(self.base_dir, 'params.json'), 'w') as f:
                    json.dump(params, f, sort_keys=False, indent=4)

            print('>> Generating agreeable and polarity dataloaders')
            params_ds_agree = params['dataset']['agree']
            params_ds_polar = params['dataset']['polar']
            dl_agree, dl_agree_val, weights_agree = self.ds.generate_agreeable(**params_ds_agree)
            dl_polar, dl_polar_val, weights_polar = self.ds.generate_polarity(**params_ds_polar)

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
                                                          save_path=os.path.join(self.base_dir, 'amodel.pt'))
                plt.savefig(os.path.join(self.base_dir, 'amodel_loss.png'))
                plt.close()
                self.update_output('amodel_loss', amodel_loss)
                self.update_output('amodel_loss_val', amodel_loss_val)

            if (self.resume and not os.path.isfile(self.path_fn('agree_val.png'))) or (
                    not self.resume and params['eval']):
                print('>> Getting train agreeable distribution')
                plt.figure()
                get_distribution_agree(self.amodel, dl_agree, -1., display=True)
                get_distribution_agree(self.amodel, dl_agree, 0., display=True)
                get_distribution_agree(self.amodel, dl_agree, 1., display=True)
                plt.savefig(os.path.join(self.base_dir, 'agree_train.png'))
                plt.close()
                print('>> Getting val agreeable distribution')
                plt.figure()
                get_distribution_agree(self.amodel, dl_agree_val, -1., display=True)
                get_distribution_agree(self.amodel, dl_agree_val, 0., display=True)
                get_distribution_agree(self.amodel, dl_agree_val, 1., display=True)
                plt.savefig(os.path.join(self.base_dir, 'agree_val.png'))
                plt.close()

            if self.resume and os.path.isfile(self.path_fn('pmodel.pt')):
                print('>> Loading PModel')
                self.pmodel = torch.load(self.path_fn('pmodel.pt'))
            else:
                print('>> Constructing PModel')
                params_arch_pmodel = params['architecture']['polar']
                self.pmodel = PModel(**params_arch_pmodel)

                print('>> Fitting PModel')
                params_pmodel = params['model']['polar']
                pmodel_loss, pmodel_loss_val = PModel.fit(self.pmodel, dl_polar, dl_polar_val, **params_pmodel,
                                                          save_path=os.path.join(self.base_dir, 'pmodel.pt'))
                plt.savefig(os.path.join(self.base_dir, 'pmodel_loss.png'))
                plt.close()
                self.update_output('pmodel_loss', pmodel_loss)
                self.update_output('pmodel_loss_val', pmodel_loss_val)

            if (self.resume and not os.path.isfile(self.path_fn('polar_val.png'))) or (
                    not self.resume and params['eval']):
                print('>> Getting train polarity distribution')
                plt.figure()
                get_distribution_polar(self.pmodel, dl_polar, 0., display=True)
                get_distribution_polar(self.pmodel, dl_polar, 1., display=True)
                plt.savefig(os.path.join(self.base_dir, 'polar_train.png'))
                plt.close()
                print('>> Getting val polarity distribution')
                plt.figure()
                get_distribution_polar(self.pmodel, dl_polar_val, 0., display=True)
                get_distribution_polar(self.pmodel, dl_polar_val, 1., display=True)
                plt.savefig(os.path.join(self.base_dir, 'polar_val.png'))
                plt.close()

            if self.resume and os.path.isfile(self.path_fn('features_y_train.npy')):
                print('>> Loading train sample variational')
                features = np.load(self.path_fn('features_X_train.npy'))
                labels = np.load(self.path_fn('features_y_train.npy'))
            else:
                print('>> Generating train sample variational')
                params_feature_train = params['feature']['train']
                features, labels = sample_variational(self.amodel, self.pmodel, self.ds.stance_train,
                                                      self.ds.headline, self.ds.body, **params_feature_train,
                                                      save_path=os.path.join(self.base_dir, 'features_{}_{}.npy'),
                                                      save_path_type='train')

            if self.resume and os.path.isfile(self.path_fn('features_y_val.npy')):
                print('>> Loading val sample variational')
                features_val = np.load(self.path_fn('features_X_val.npy'))
                labels_val = np.load(self.path_fn('features_y_val.npy'))
            else:
                print('>> Generating val sample variational')
                params_feature_val = params['feature']['val']
                features_val, labels_val = sample_variational(self.amodel, self.pmodel, self.ds.stance_val,
                                                              self.ds.headline, self.ds.body, **params_feature_val,
                                                              save_path=os.path.join(self.base_dir,
                                                                                     'features_{}_{}.npy'),
                                                              save_path_type='val', save_sd_vec=False,
                                                              load_sd_vec=os.path.join(self.base_dir,
                                                                                       'features_sd_vec.npy'))

            if not self.resume or not os.path.isfile(self.path_fn('clf.pkl')):
                print('>> Fitting XGBoost classifier')
                self.clf.load_datasets(features, labels, features_val, labels_val)
                evals_result = self.clf.fit()
                self.clf.save(os.path.join(self.base_dir, 'clf.pkl'))
                self.update_output('evals_result', evals_result)
            else:
                self.clf.load(self.path_fn('clf.pkl'))

            print('>> Predicting test dataset and saving')
            params_test = params['test']['pred']
            pred = self.test_ds.predict(self.amodel, self.pmodel, self.clf,
                                        os.path.join(self.base_dir, 'features_sd_vec.npy'),
                                        **params_test)
            self.update_output('pred', pred)
            self.test_ds.submit(self.base_dir)

            score, _ = score_submission(labels, self.clf.predict(features))
            score_val, _ = score_submission(labels_val, self.clf.predict(features_val))
            score_test, _ = score_submission([STANCE_TO_LABEL[x] for x in self.true_test_label],
                                             [STANCE_TO_LABEL[x] for x in pred])
            self.update_output('score', score)
            self.update_output('score_val', score_val)
            self.update_output('score_test', score_test)

            print(f'>> Training score = {score} | val score = {score_val} | test score = {score_test}')

            with open(os.path.join(self.base_dir, f'score_{round(score_test)}.json'), 'w') as f:
                all_scores = {'score': score, 'score_val': score_val, 'score_test': score_test}
                json.dump(all_scores, f, sort_keys=False, indent=4)

            end_time = time.time()
            print(f'>> Execution took {round((end_time - start_time) / 60, 2)} minutes')
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
