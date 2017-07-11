#!/usr/bin/env python3
import argparse
from functools import partial
from pathlib import Path
import multiprocessing
from typing import Tuple, Optional
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

import dataset
import utils


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('predictions', nargs='+')
    arg('--output')
    arg('--recalibrate', type=int, default=1)
    arg('--merge', default='mean', choices=['mean', 'gmean', 'vote'])
    arg('--verbose', action='store_true')
    args = parser.parse_args()

    test_predictions, valid_predictions, valid_f2s = [], [], []
    with multiprocessing.Pool() as pool:
        for filename, (test_pred, valid_pred, valid_f2) in zip(
                args.predictions,
                pool.map(partial(load_recalibrate_prediction,
                                 recalibrate=args.recalibrate,
                                 verbose=args.verbose),
                         map(Path, args.predictions))):
            if args.output:
                test_predictions.append(test_pred)
            valid_predictions.append(valid_pred)
            valid_f2s.append(valid_f2)
            print('{:.5f} valid F2 for {}'.format(valid_f2, filename))
    print('Mean F2 score: {:.5f}'.format(np.mean(valid_f2s)))
    valid_prediction = merge_predictions(valid_predictions, args.merge)
    print('Final valid F2 after {} blend: {:.5f}'.format(
        args.merge, dataset.f2_score(get_true_data(valid_prediction),
                                     valid_prediction.as_matrix())))
    if not args.output:
        return

    prediction = merge_predictions(test_predictions, args.merge)
    out_df = pd.DataFrame([
        {'image_name': image_name,
         'tags': ' '.join(c for c in prediction.columns if row[c])
         } for image_name, row in prediction.iterrows()])
    out_df.set_index('image_name', inplace=True)
    sample_submission = pd.read_csv(
        utils.DATA_ROOT / 'sample_submission_v2.csv', index_col=0)
    assert set(sample_submission.index) == set(out_df.index)
    out_df = out_df.loc[sample_submission.index]
    out_df.to_csv(args.output, index_label='image_name')
    print('Saved submission to {}'.format(args.output))


def merge_predictions(predictions, merge_mode):
    prediction = pd.concat(predictions)
    if merge_mode == 'vote':
        return (prediction > THRESHOLD).groupby(level=0).median()
    elif merge_mode == 'mean':
        return utils.mean_df(prediction) > THRESHOLD
    elif merge_mode == 'gmean':
        return utils.gmean_df(prediction) > THRESHOLD


THRESHOLD = 0.2


def load_recalibrate_prediction(
        path: Path, recalibrate: bool, verbose=False,
        ) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, float]:
    prefix, aug_kind = path.stem.split('_', 1)
    assert prefix in {'test', 'val'}
    valid_df = pd.read_hdf(
        path.parent / 'val_{}.h5'.format(aug_kind), index_col=0)
    valid_data = valid_df.as_matrix()
    true_data = get_true_data(valid_df)

    if recalibrate:
        f2_score = dataset.f2_score(
            true_data, valid_df.as_matrix() > THRESHOLD)
        print('{} with default threshold: {:.5f}'.format(path, f2_score))
        thresholds, f2 = optimise_f2_thresholds(true_data, valid_data)
        if verbose:
            print('Train F2 {:.5f}, thresholds: {}'.format(f2, thresholds))
        valid_dfs = []
        valid_df_logits = logit(valid_df)
        for train_ids, valid_ids in (
                KFold(n_splits=5, shuffle=True).split(true_data)):
            fold_thresholds, f2 = optimise_f2_thresholds(
                true_data[train_ids], valid_data[train_ids])
            valid_dfs.append(sigmoid(
                recalibrated_logits(valid_df_logits.iloc[valid_ids],
                                    fold_thresholds)))
            if verbose:
                print('Train F2: {:.5f}, valid F2 {:.5f}, thresholds {}'
                      .format(f2, dataset.f2_score(true_data[valid_ids],
                                                   valid_dfs[-1] > THRESHOLD),
                              fold_thresholds))
        valid_df = pd.concat(valid_dfs).loc[valid_df.index]
    f2_score = dataset.f2_score(true_data, valid_df.as_matrix() > THRESHOLD)

    if path.exists() and prefix != 'val':
        test_df = pd.read_hdf(path, index_col=0).groupby(level=0).mean()
        test_df_logits = logit(test_df)
        if recalibrate:
            test_df_logits = recalibrated_logits(test_df_logits, thresholds)
        test_df = sigmoid(test_df_logits)
    else:
        test_df = None
    return test_df, valid_df, f2_score


def get_true_data(df):
    true_df = pd.read_csv(utils.DATA_ROOT / 'train_v2.csv', index_col=0)
    true_data = np.zeros((len(df), dataset.N_CLASSES), dtype=np.uint8)
    for i, tags in enumerate(true_df.loc[df.index]['tags']):
        for tag in tags.split():
            true_data[i, dataset.CLASSES.index(tag)] = 1
    return true_data


def recalibrated_logits(df, thresholds):
    df = df.copy()
    for column, threshold in zip(df.columns, thresholds):
        # logit(threshold) -> logit(THRESHOLD)
        df[column] += logit(THRESHOLD) - logit(threshold)
    return df


def logit(p):
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def optimise_f2_thresholds(y, p, verbose=False, resolution=100):
    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(dataset.N_CLASSES):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        # TODO - calculate only one column
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            score = dataset.f2_score(y, p2)
        return score

    x = [THRESHOLD] * dataset.N_CLASSES
    best_score = 0
    for i, cls in enumerate(dataset.CLASSES):
        if cls in dataset.RARE_CLASSES:
            continue
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 /= resolution
            x[i] = i2
            score = mf(x)
            if score > best_score:
                best_i2 = i2
                best_score = score
        x[i] = best_i2
        if verbose:
            print(i, best_i2, best_score)

    return x, best_score


if __name__ == '__main__':
    main()
