import os
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AlbertTokenizer, AlbertModel

STANCE_TO_LABEL = {
    'agree': 0,
    'disagree': 1,
    'discuss': 2,
    'unrelated': 3
}

LABEL_TO_STANCE = {v: k for k, v in STANCE_TO_LABEL.items()}

STANCE_TO_AGREE = {
    'agree': 1.,
    'disagree': -1.,
    'discuss': 1.,
    'unrelated': 0.
}
STANCE_TO_POLARITY = {
    'agree': 1,
    'disagree': 1,
    'discuss': 0,
    'unrelated': np.nan
}


class FNCDataset:

    def __init__(self, bodies_fp, stance_fp,
                 body_npy='models/body.npy', headline_npy='models/headline.npy', stance_df_pkl='models/stance_df.pkl',
                 stance_split_format='models/stance_{}.pkl', train_ratio=0.8, seed=None, force_recompute=False):

        self.body_npy = body_npy
        self.headline_npy = headline_npy
        self.stance_df_pkl = stance_df_pkl

        self.bodies_df = pd.read_csv(bodies_fp)

        if os.path.isfile(stance_df_pkl):
            print(f'Loading stance df from {stance_df_pkl}')
            self.stance_df = pd.read_pickle(stance_df_pkl)
        else:
            self.stance_df = pd.read_csv(stance_fp)

        if os.path.isfile(headline_npy):
            print(f'Loading headline matrix from {headline_npy}')
            self.headline = np.load(headline_npy)
        else:
            self.headline = None

        if os.path.isfile(body_npy):
            print(f'Loading body matrix from {body_npy}')
            self.body = np.load(body_npy)
        else:
            self.body = None

        self.id_mapping = None
        self.stance_train = None
        self.stance_val = None

        self._vectorize_bodies()
        self._labelize_stance()
        self._headline_id()
        self._train_val_split(self.stance_df, train_ratio=train_ratio, seed=seed,
                              save_path_format=stance_split_format, force_recompute=force_recompute)

        if not os.path.isfile(stance_df_pkl):
            self.stance_df.to_pickle(stance_df_pkl)

    def generate_agreeable(self, batch_size=32, augment_dict=dict(), batch_size_val=128, verbose=False):
        dataset = AgreeDataset(self.stance_train, self.body, self.headline)
        dataset_val = AgreeDataset(self.stance_val, self.body, self.headline)

        labels, frequency = np.unique(dataset.label, return_counts=True)
        if verbose:
            print(f'Agreeableness label distribution:\n{labels}\n{frequency}')

        weights = np.zeros_like(dataset.label)
        for label, freq in zip(labels, frequency):
            if label in augment_dict:
                weights[dataset.label == label] = (1. + augment_dict[label]) / freq
            else:
                weights[dataset.label == label] = 1. / freq
        weights = weights / weights.sum()

        class_weights = 1. / frequency
        class_weights = class_weights / class_weights.sum()

        sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(weights))
        return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size), \
               torch.utils.data.DataLoader(dataset_val, batch_size=batch_size_val), \
               class_weights.astype(np.float32)

    def generate_polarity(self, batch_size=32, augment_dict=dict(), batch_size_val=128, verbose=False):
        dataset = PolarDataset(self.stance_train, self.body)
        dataset_val = PolarDataset(self.stance_val, self.body)

        labels, frequency = np.unique(dataset.label, return_counts=True)
        if verbose:
            print(f'Polarity label distribution:\n{labels}\n{frequency}')

        weights = np.zeros_like(dataset.label)
        for label, freq in zip(labels, frequency):
            if label in augment_dict:
                weights[dataset.label == label] = (1. + augment_dict[label]) / freq
            else:
                weights[dataset.label == label] = 1. / freq
        weights = weights / weights.sum()

        class_weights = 1. / frequency
        class_weights = class_weights / class_weights.sum()

        sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(weights))
        return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size), \
               torch.utils.data.DataLoader(dataset_val, batch_size=batch_size_val), \
               class_weights.astype(np.float32)

    def _vectorize_bodies(self):

        mapping = dict()
        for idx, body_id in enumerate(self.bodies_df['Body ID']):
            mapping[body_id] = idx
        self.id_mapping = mapping

        if self.body is not None:
            return

        model = AlbertModel.from_pretrained("albert-base-v2")
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

        body_matrix = np.zeros((len(self.bodies_df), 768))
        print('Generating body matrix...')

        for idx, body in enumerate(tqdm(self.bodies_df['articleBody'])):
            encoding = tokenizer(body, return_tensors='pt', truncation=True, padding=True)
            output = model(**encoding)
            body_matrix[idx] = output.pooler_output.detach().numpy()

        self.bodies_df['features'] = list(body_matrix)
        self.body = body_matrix

        np.save(self.body_npy, body_matrix)

        mapped_ids = list()
        for body_id in self.stance_df['Body ID']:
            mapped_ids.append(mapping[body_id])
        self.stance_df['Mapped Body ID'] = mapped_ids

    def _labelize_stance(self):
        if isinstance(self.stance_df['Stance'].iloc[0], str):
            self.stance_df['Agreeableness'] = self.stance_df['Stance'].apply(lambda x: STANCE_TO_AGREE[x])
            self.stance_df['Polarity'] = self.stance_df['Stance'].apply(lambda x: STANCE_TO_POLARITY[x])
            self.stance_df['Stance'] = self.stance_df['Stance'].apply(lambda x: STANCE_TO_LABEL[x])

    def _headline_id(self):
        if self.headline is not None:
            return

        model = AlbertModel.from_pretrained("albert-base-v2")
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

        row_id = 0
        headline_dict = dict()
        headline_ids = np.zeros(len(self.stance_df), dtype=np.uint16)

        for idx, h in enumerate(self.stance_df['Headline']):
            if h in headline_dict:
                headline_ids[idx] = headline_dict[h]
            else:
                headline_dict[h] = row_id
                headline_ids[idx] = row_id
                row_id += 1

        self.stance_df['Headline ID'] = headline_ids

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

    def _train_val_split(self, df, train_ratio, save_path_format, seed=None, force_recompute=False):

        if os.path.isfile(save_path_format.format('train')) and os.path.isfile(
                save_path_format.format('val')) and not force_recompute:
            print(f'Loading train and val stance df from {save_path_format.format("train")} and '
                  f'{save_path_format.format("test")}')
            self.stance_train = pd.read_pickle(save_path_format.format('train'))
            self.stance_val = pd.read_pickle(save_path_format.format('val'))
            return

        print('Computing train and val stance df')

        if seed is not None:
            np.random.seed(seed)

        stance = df['Stance'].values
        label_kinds = np.unique(stance)
        idx_range = np.arange(len(df))

        train_indices = list()
        val_indices = list()

        for label in label_kinds:
            index = idx_range[stance == label]
            train_selector = np.zeros_like(index, dtype=bool)
            train_selector[
                np.random.choice(len(index), size=int(train_ratio * len(index)), replace=False)] = True

            train_indices += index[train_selector].tolist()
            val_indices += index[~train_selector].tolist()

        train_indices = sorted(train_indices)
        val_indices = sorted(val_indices)

        self.stance_train = df.iloc[train_indices, :]
        self.stance_val = df.iloc[val_indices, :]

        self.stance_train.to_pickle(save_path_format.format('train'))
        self.stance_val.to_pickle(save_path_format.format('val'))


class AgreeDataset(torch.utils.data.Dataset):
    def __init__(self, df, body_matrix, headline_matrix):
        self.body_matrix = body_matrix
        self.headline_matrix = headline_matrix

        self.headline_id = df['Headline ID'].values
        self.body_id = df['Mapped Body ID'].values
        self.label = df['Agreeableness'].values

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return (self.headline_matrix[self.headline_id[item]], self.body_matrix[self.body_id[item]]), self.label[item]


class PolarDataset(torch.utils.data.Dataset):
    def __init__(self, df, body_matrix):
        self.body_matrix = body_matrix

        df = df.dropna(subset=['Polarity'])

        self.body_id = df['Mapped Body ID'].values
        self.label = df['Polarity'].values

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.body_matrix[self.body_id[item]], self.label[item]


if __name__ == '__main__':
    BASE_DIR = 'data'
    STANCE = 'train_stances.csv'
    BODIES = 'train_bodies.csv'

    ds = FNCDataset(os.path.join(BASE_DIR, BODIES), os.path.join(BASE_DIR, STANCE))
