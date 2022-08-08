import numpy as np
import xgboost as xgb

from pickle import dump, load


class Classifier:
    def __init__(self, **kwargs):
        self.clf = xgb.XGBClassifier(objective='multi:softmax', **kwargs)
        self.default_params = {
            'objective': 'multi:softmax',
            'eval_metric': 'mlogloss',
            'num_class': 4
        }
        # https://xgboost.readthedocs.io/en/stable/python/python_api.html
        self.train_dataset = None
        self.val_dataset = None

    def load_datasets(self, features, labels, features_val, labels_val):
        self.train_dataset = xgb.DMatrix(data=features, label=labels)
        self.val_dataset = xgb.DMatrix(data=features_val, label=labels_val)
        # self.train_dataset = (features, labels)
        # self.val_dataset = (features_val, labels_val)

        nans = np.isnan(features).sum(axis=0)
        if nans.sum() > 0:
            print('NaNs detected in training data')
            print(nans)

        nans_val = np.isnan(features_val).sum(axis=0)
        if nans_val.sum() > 0:
            print('NaNs detected in val data')
            print(nans_val)

        infs = np.isinf(features).sum(axis=0)
        if infs.sum() > 0:
            print('Infs detected in training data')
            print(infs)

        infs_val = np.isinf(features_val).sum(axis=0)
        if infs_val.sum() > 0:
            print('Infs detected in val data')
            print(infs_val)

    def load(self, load_fp):
        if load_fp[-4:] != '.pkl':
            load_fp += '.pkl'

        print(f'Loading classifier from {load_fp}')
        with open(load_fp, 'rb') as f:
            self.clf = load(f)

    def fit(self, params={}, **kwargs):
        assert self.train_dataset is not None, 'Datasets have not been loaded'

        evals_result = dict()
        self.clf = xgb.train(params={**self.default_params, **params},
                             dtrain=self.train_dataset,
                             num_boost_round=40,
                             evals=[(self.train_dataset, 'train'), (self.val_dataset, 'val')],
                             early_stopping_rounds=5,
                             evals_result=evals_result, verbose_eval=False)

        # features, labels = self.train_dataset
        # values, frequency = np.unique(labels, return_counts=True)
        # weights = np.zeros_like(labels, dtype=np.float64)
        # for v, freq in zip(values, frequency):
        #     weights[labels == v] = 1. / freq
        # weights = weights / weights.sum()

        # self.clf.fit(features, labels, sample_weight=weights)

        return evals_result

    def predict(self, X, **kwargs):
        # return self.clf.predict(X, **kwargs)
        return self.clf.predict(xgb.DMatrix(data=X), **kwargs)

    def save(self, save_fp):
        if save_fp[-4:] != '.pkl':
            save_fp += '.pkl'

        print(f'Saving classifier to {save_fp}')
        with open(save_fp, 'wb') as f:
            dump(self.clf, f)


if __name__ == '__main__':
    clf = Classifier()
    X = np.random.normal(0, 1, size=(100, 5))
    y = np.random.choice(4, size=(100,))

    clf.load_datasets(X, y, X, y)
    res = clf.fit()
    pred = clf.predict(X)
