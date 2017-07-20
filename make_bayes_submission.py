#!/usr/bin/env python3
import argparse
import itertools
from functools import partial
from pathlib import Path
import multiprocessing
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import tqdm

import dataset
import utils
from make_submission import get_true_data


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('predictions', nargs='+')
    arg('--output')
    arg('--merge', default='mean', choices=['mean', 'gmean'])
    args = parser.parse_args()

    test_predictions, valid_predictions, valid_f2s = [], [], []
    for filename in args.predictions:
        test_pred, valid_pred, valid_f2 = load_prediction(Path(filename))
        if args.output:
            test_predictions.append(test_pred)
        valid_f2s.append(valid_f2)
        valid_predictions.append(valid_pred)
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


def load_prediction(path: Path
                    ) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, float]:
    prefix, aug_kind = path.stem.split('_', 1)
    assert prefix in {'test', 'val'}
    valid_df = pd.read_hdf(
        path.parent / 'val_{}.h5'.format(aug_kind), index_col=0)
    true_data = get_true_data(valid_df)

    threshold = 0.2
    f2_score = dataset.f2_score(true_data, valid_df.as_matrix() > threshold)

    # valid_predictions = get_df_prediction(valid_df)
    # f2_score = dataset.f2_score(true_data, valid_predictions.as_matrix())

    if path.exists() and prefix != 'val':
        test_df = pd.read_hdf(path, index_col=0).groupby(level=0).mean()
    else:
        test_df = None
    return test_df, valid_df, f2_score


def merge_predictions(predictions, merge_mode):
    prediction = pd.concat(predictions)  # type: pd.DataFrame
    if merge_mode == 'mean':
        df = utils.mean_df(prediction)
    elif merge_mode == 'gmean':
        df = utils.gmean_df(prediction)
    else:
        raise ValueError('Unexpected merge_mode')
    return get_df_prediction(df)


def get_df_prediction(df: pd.DataFrame, min_p=0.05, max_p=0.5, add_hacks=False,
                      ) -> pd.DataFrame:
    with multiprocessing.Pool() as pool:
        fn = partial(get_item_prediction,
                     min_p=min_p, max_p=max_p, add_hacks=add_hacks)
        data = pool.map(fn, [item for _, item in df.iterrows()])
    return pd.DataFrame(data=data, index=df.index, columns=df.columns)


def get_item_prediction(item, min_p, max_p, add_hacks):
    free_vars, value = get_free_vars(item, min_p, max_p)
    if not free_vars:
        return value
    ys = get_true_candidates(item, free_vars, value, add_hacks=add_hacks)
    scores = []
    for var_set in itertools.product(*free_vars):
        for i, v in var_set:
            value[i] = v
        score = 0
        for y, prob in ys:
            score += prob * f2_fast(y, value)
        scores.append((score, value.copy()))
    score, value = max(scores, key=lambda x: x[0])
    return value


def get_free_vars(item, min_p, max_p):
    free_vars = []
    value = np.zeros(dataset.N_CLASSES)
    for i, p in enumerate(item):
        if p >= min_p:
            if p >= max_p:
                value[i] = 1
            else:
                free_vars.append([(i, 0), (i, 1)])
    return free_vars, value


def get_true_candidates(item, free_vars, value, add_hacks):
    ys = []
    for var_set in itertools.product(*free_vars):
        for i, v in var_set:
            value[i] = v
        if value.sum() == 0:
            continue
        prob = 1.0
        if add_hacks:
            if value[dataset.CLOUDY_ID] and value.sum() > 1:
                # cloudy + something = 0
                prob *= 0.9
            elif sum(value[i] for i in dataset.WEATHER_IDS) > 1:
                # several weather tags
                prob *= 0.9
        for v, p in zip(value, item):
            if v == 0:
                p = 1 - p
            prob *= p
        if prob > 1e-7:
            ys.append((value.copy(), prob))
    return ys


def f2_fast(y_true, y_pred):
    ''' F2 for a single row.
    '''
    tp = y_true @ y_pred
    r = tp / y_true.sum()
    if r == 0:
        return 0
    p = tp / (y_pred.sum() + 1e-6)
    beta2 = 4
    return (1 + beta2) * p * r / (beta2 * p + r)


if __name__ == '__main__':
    main()
