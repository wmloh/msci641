import sys
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version

from dataset import STANCE_TO_LABEL


def generate_features(stances, df, dataset, name, unlabelled=False, test=False):
    h, b, y = [], [], []

    if test:
        for stance in stances:
            h.append(stance['Headline'])
            b.append(dataset.articles[stance['Body ID']])
    else:
        h = df['Headline'].to_list()
        if unlabelled:
            # for stance in stances:
            for bid in df['Body ID']:
                # h.append(stance['Headline'])
                b.append(dataset.articles[bid])

            assert len(h) == len(b)

        else:
            y = df['Stance'].to_list()
            for bid in df['Body ID']:
                # y.append(STANCE_TO_LABEL[stance['Stance']])
                # h.append(stance['Headline'])
                b.append(dataset.articles[bid])

            assert len(h) == len(b) == len(y)

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap." + name + ".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting." + name + ".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity." + name + ".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand." + name + ".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    if unlabelled:
        return X
    return X, y


if __name__ == '__main__':
    ds = DataSet(path='data')
    df_train = pd.read_pickle('models/stance_train.pkl')
    df_val = pd.read_pickle('models/stance_val.pkl')

    X_train, y_train = generate_features(ds.stances, df_train, ds, 'train')
    X_val, y_val = generate_features(ds.stances, df_val, ds, 'val')

    competition_dataset = DataSet("competition_test", path='data')
    X_competition, y_competition = generate_features(competition_dataset.stances, None, competition_dataset,
                                                     "competition", test=True)

s = False
if s:
    check_version()
    parse_params()

    # Load the training dataset and generate folds
    d = DataSet()
    folds, hold_out = kfold_split(d, n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d, folds, hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout, y_holdout = generate_features(hold_out_stances, d, "holdout")
    for fold in fold_stances:
        Xs[fold], ys[fold] = generate_features(fold_stances[fold], d, str(fold))

    best_score = 0
    best_fold = None

    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score / max_fold_score

        print("Score for fold " + str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf

    dump(best_fold, "best_clf.joblib")

    # Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual, predicted)
    print("")
    print("")

    # Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual, predicted)
