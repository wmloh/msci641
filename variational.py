import os
import numpy as np
import torch

from tqdm import tqdm, trange
from scipy.stats import skew

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_variational(amodel, pmodel,
                       df, headline_matrix, body_matrix,
                       sd_factor, size,
                       batch_size, save_path=None, save_path_type=None, save_sd_vec=True,
                       load_sd_vec=None):
    amodel.to(DEVICE)
    pmodel.to(DEVICE)

    headline_ids = df['Headline ID'].values
    body_ids = df['Mapped Body ID'].values

    if load_sd_vec is not None:
        print(f"Loading sd_vec from {load_sd_vec}")
        sd_vec = np.load(load_sd_vec)
    else:
        print(f"Generating new sd_vec")
        sd_vec = body_matrix.std(axis=0) * sd_factor
    features = np.zeros(shape=(len(headline_ids), 6))

    with torch.inference_mode():
        for idx in trange(int(np.ceil(len(headline_ids) / batch_size))):
            headline = headline_matrix[headline_ids[idx * batch_size: (idx + 1) * batch_size]]
            body = body_matrix[body_ids[idx * batch_size: (idx + 1) * batch_size]]

            num_samples = len(headline)

            headlines = list()
            bodies = list()

            for i, (h, b) in enumerate(zip(headline, body)):
                headlines.append(np.broadcast_to(h, (size, 768)))
                bodies.append(np.broadcast_to(b, (size, 768)))

            headlines = np.vstack(headlines)
            bodies = np.vstack(bodies)
            bodies += np.random.normal(0, sd_vec, size=bodies.shape)

            headlines = torch.from_numpy(headlines).float().to(DEVICE)
            bodies = torch.from_numpy(bodies).float().to(DEVICE)

            agreeableness = amodel.predict(headlines, bodies).reshape(num_samples, size).cpu().numpy()
            polarity = pmodel(bodies).reshape(num_samples, size).cpu().numpy()

            mu_a = agreeableness.mean(axis=1)
            sigma_a = agreeableness.std(axis=1)
            g_a = skew(agreeableness, axis=1)

            mu_p = polarity.mean(axis=1)
            sigma_p = polarity.std(axis=1)
            g_p = skew(polarity, axis=1)

            if np.isnan(g_p).sum() > 0:
                g_p[np.isnan(g_p)] = 0
            if np.isinf(g_p).sum() > 0:
                g_p[np.isinf(g_p)] = 0
            if np.isnan(g_a).sum() > 0:
                g_a[np.isnan(g_a)] = 0
            if np.isinf(g_a).sum() > 0:
                g_a[np.isinf(g_a)] = 0

            features[idx * batch_size: (idx + 1) * batch_size] = \
                np.stack([mu_a, sigma_a, g_a, mu_p, sigma_p, g_p], axis=1)

    if 'Stance' in df:
        labels = df['Stance'].values
    else:
        labels = None

    if save_path is not None:
        np.save(save_path.format('X', save_path_type), features)
        if save_sd_vec:
            print(f"Saving sd_vec to {save_path.format('sd', 'vec')}")
            np.save(save_path.format('sd', 'vec'), sd_vec)
        if 'Stance' in df:
            np.save(save_path.format('y', save_path_type), labels)

    amodel.cpu()
    pmodel.cpu()

    return features, labels
