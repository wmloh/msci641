import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import trange
from scipy.stats import skew

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_distribution_agree(model, dl, label, display=False, batch_size=128):
    model.to(DEVICE)
    ds = dl.dataset
    index = ds.label == label

    headline = ds.headline_matrix[ds.headline_id[index]]
    body = ds.body_matrix[ds.body_id[index]]

    result = list()

    with torch.inference_mode():
        for idx in trange(int(np.ceil(len(headline) / batch_size))):
            headline_batch = torch.from_numpy(headline[idx * batch_size: (idx + 1) * batch_size]).float().to(DEVICE)
            body_batch = torch.from_numpy(body[idx * batch_size: (idx + 1) * batch_size]).float().to(DEVICE)
            result += model.predict(headline_batch, body_batch).tolist()

    result = sorted(result)
    mean = float('%.3g' % np.mean(result))
    sd = float('%.3g' % np.std(result))
    skewness = float('%.3g' % skew(result))

    if display:
        plt.hist(result, alpha=0.4, bins='doane',
                 label=f'[{int(label)}]: $\\mu$={mean}, $\\sigma$={sd}, $g$={skewness}')
        plt.ylabel('Frequency')
        plt.xlabel('Predicted value')
        plt.legend()

    model.cpu()

    return result


def get_distribution_polar(model, dl, label, display=False, batch_size=128):
    model.to(DEVICE)
    ds = dl.dataset
    index = ds.label == label

    body = ds.body_matrix[ds.body_id[index]]

    result = list()

    with torch.inference_mode():
        for idx in trange(int(np.ceil(len(body) / batch_size))):
            body_batch = torch.from_numpy(body[idx * batch_size: (idx + 1) * batch_size]).float().to(DEVICE)
            result += model(body_batch).tolist()

    result = sorted(result)
    mean = float('%.3g' % np.mean(result))
    sd = float('%.3g' % np.std(result))
    skewness = float('%.3g' % skew(result))

    if display:
        plt.hist(result, alpha=0.4, bins='auto',
                 label=f'[{int(label)}]: $\\mu$={mean}, $\\sigma$={sd}, $g$={skewness}')
        plt.ylabel('Frequency')
        plt.xlabel('Predicted value')
        plt.legend()

    model.cpu()

    return result
