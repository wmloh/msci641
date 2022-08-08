import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AlbertTokenizer, AlbertModel
from dataset import STANCE_TO_LABEL, LABEL_TO_STANCE
from variational import sample_variational

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset:
    def __init__(self, test_body_fp, test_csv_fp,
                 body_npy='models/test_body.npy', headline_npy='models/test_headline.npy',
                 csv_pkl='models/test_csv.pkl'):

        self.body_npy = body_npy
        self.headline_npy = headline_npy
        self.csv_pkl = csv_pkl

        self.test_bodies = pd.read_csv(test_body_fp)

        if os.path.isfile(csv_pkl):
            print(f'Loading test csv from {csv_pkl}')
            self.test_csv = pd.read_pickle(csv_pkl)
        else:
            self.test_csv = pd.read_csv(test_csv_fp)

        if os.path.isfile(headline_npy):
            print(f'Loading test headline matrix from {headline_npy}')
            self.headline = np.load(headline_npy)
        else:
            self.headline = None

        if os.path.isfile(body_npy):
            print(f'Loading test body matrix from {body_npy}')
            self.body = np.load(body_npy)
        else:
            self.body = None

        self.id_mapping = None
        self.last_pred = None
        self.last_features = None

        self._vectorize_bodies()
        self._headline_id()

        if not os.path.isfile(csv_pkl):
            self.test_csv.to_pickle(csv_pkl)

    def _vectorize_bodies(self):
        mapping = dict()
        for idx, body_id in enumerate(self.test_bodies['Body ID']):
            mapping[body_id] = idx
        self.id_mapping = mapping

        if self.body is not None:
            return

        model = AlbertModel.from_pretrained("albert-base-v2")
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

        body_matrix = np.zeros((len(self.test_bodies), 768))
        print('Generating body matrix...')

        for idx, body in enumerate(tqdm(self.test_bodies['articleBody'])):
            encoding = tokenizer(body, return_tensors='pt', truncation=True, padding=True)
            output = model(**encoding)
            body_matrix[idx] = output.pooler_output.detach().numpy()

        self.body = body_matrix
        np.save(self.body_npy, body_matrix)

        mapped_ids = list()
        for body_id in self.test_csv['Body ID']:
            mapped_ids.append(mapping[body_id])
        self.test_csv['Mapped Body ID'] = mapped_ids

    def _headline_id(self):
        if self.headline is not None:
            return

        model = AlbertModel.from_pretrained("albert-base-v2")
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

        row_id = 0
        headline_dict = dict()
        headline_ids = np.zeros(len(self.test_csv), dtype=np.uint16)

        for idx, h in enumerate(self.test_csv['Headline']):
            if h in headline_dict:
                headline_ids[idx] = headline_dict[h]
            else:
                headline_dict[h] = row_id
                headline_ids[idx] = row_id
                row_id += 1

        self.test_csv['Headline ID'] = headline_ids

        ids = list(headline_dict.values())
        headlines = list(headline_dict.keys())
        headline_df = pd.DataFrame(data={'Headline ID': ids, 'Headline': headlines})
        headline_df.set_index('Headline ID', inplace=True)

        headline_matrix = np.zeros((len(headline_df), 768))

        print('Generating headline matrix...')
        for idx, body in enumerate(tqdm(headline_df['Headline'])):
            encoding = tokenizer(body, return_tensors='pt', truncation=True, padding=True)
            output = model(**encoding)
            headline_matrix[idx] = output.pooler_output.detach().numpy()

        self.headline = headline_matrix
        np.save(self.headline_npy, headline_matrix)

    def generate_features(self, batch_size=64):
        dataset = FeatureDataset(self.test_csv, self.body, self.headline)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    def predict(self, amodel, pmodel, clf, X_base, batch_size=64,
                sd_factor=0.07, variational_size=8, v_batch_size=16):
        dl = self.generate_features(batch_size=batch_size)

        amodel.to(DEVICE)
        pred = list()
        with torch.inference_mode():
            for headline, body in dl:
                headline = headline.float().to(DEVICE)
                body = body.float().to(DEVICE)

                pred += amodel.predict_proba(headline, body).cpu().tolist()

        pred = np.asarray(pred)
        features = np.hstack((pred, X_base))

        amodel.cpu()

        pred = clf.predict(features)
        pred = np.array([LABEL_TO_STANCE[x] for x in pred])
        self.last_pred = pred
        self.last_features = features

        return pred

    def submit(self, dir_path):
        assert self.last_pred is not None, 'The \'pred\' function must be called'

        df = self.test_csv[['Headline', 'Body ID']].copy()
        df['Stance'] = self.last_pred
        df.to_csv(os.path.join(dir_path, 'answer.csv'), index=False, encoding='utf-8')

        zf = zipfile.ZipFile(os.path.join(dir_path, 'answer.zip'), 'w', zipfile.ZIP_DEFLATED)
        zf.write(os.path.join(dir_path, 'answer.csv'), arcname='answer.csv')
        zf.close()


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, df, body_matrix, headline_matrix):
        self.body_matrix = body_matrix
        self.headline_matrix = headline_matrix

        self.headline_id = df['Headline ID'].values
        self.body_id = df['Mapped Body ID'].values

    def __len__(self):
        return len(self.headline_id)

    def __getitem__(self, item):
        return self.headline_matrix[self.headline_id[item]], self.body_matrix[self.body_id[item]]
