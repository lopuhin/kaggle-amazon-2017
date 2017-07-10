#!/usr/bin/env python3
import argparse
from functools import partial
from pathlib import Path
import multiprocessing
from typing import Tuple, Optional
import warnings

import pandas as pd
import numpy as np

import dataset
import utils


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('predictions', nargs='+')
    arg('--output')
    arg('--recalibrate', type=int, default=1)
    arg('--vote', action='store_true')
    args = parser.parse_args()

    predictions, f2s = [], []
    with multiprocessing.Pool() as pool:
        for filename, (pred, f2) in zip(
                args.predictions,
                pool.map(partial(load_recalibrate_prediction,
                                 recalibrate=args.recalibrate),
                         map(Path, args.predictions))):
            if args.output:
                predictions.append(pred)
            f2s.append(f2)
            print('{:.5f} {}'.format(f2, filename))
    print('Mean score: {:.5f}'.format(np.mean(f2s)))
    if not args.output:
        return

    prediction = pd.concat(predictions)
    if args.vote:
        prediction = (prediction > THRESHOLD).groupby(level=0).median()
    else:
        prediction = prediction.groupby(level=0).mean() > THRESHOLD
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


THRESHOLD = 0.2


def load_recalibrate_prediction(path: Path, recalibrate: bool,
                                ) -> Tuple[Optional[pd.DataFrame], float]:
    prefix, aug_kind = path.stem.split('_', 1)
    assert prefix in {'test', 'val'}
    valid_df = pd.read_hdf(
        path.parent / 'val_{}.h5'.format(aug_kind), index_col=0)
    valid_data = valid_df.as_matrix()

    true_df = pd.read_csv(utils.DATA_ROOT / 'train_v2.csv', index_col=0)
    true_data = np.zeros((len(valid_df), dataset.N_CLASSES), dtype=np.uint8)
    for i, tags in enumerate(true_df.loc[valid_df.index]['tags']):
        for tag in tags.split():
            true_data[i, dataset.CLASSES.index(tag)] = 1

    if recalibrate:
        valid_df_logits = logit(valid_df)
        f2_score = dataset.f2_score(
            true_data, valid_df.as_matrix() > THRESHOLD)
        print('{} with default threshold: {:.5f}'.format(path, f2_score))
        thresholds = optimise_f2_thresholds(true_data, valid_data)
        print('{} thresholds: {}'.format(path, thresholds))
        valid_df_logits = recalibrated_logits(valid_df_logits, thresholds)
        valid_df = sigmoid(valid_df_logits)  # type: pd.DataFrame
    f2_score = dataset.f2_score(true_data, valid_df.as_matrix() > THRESHOLD)

    if path.exists() and prefix != 'val':
        test_df = pd.read_hdf(path, index_col=0).groupby(level=0).mean()
        test_df_logits = logit(test_df)
        if recalibrate:
            test_df_logits = recalibrated_logits(test_df_logits, thresholds)
        test_df = sigmoid(test_df_logits)
    else:
        test_df = None
    return test_df, f2_score


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

    return x


if __name__ == '__main__':
    main()
