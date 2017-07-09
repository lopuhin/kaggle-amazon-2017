#!/usr/bin/env python3
import argparse
import itertools
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import tqdm

import dataset
import utils


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('predictions', nargs='+')
    arg('output')
    args = parser.parse_args()

    predictions, f2s = [], []
    for filename in args.predictions:
        pred, f2 = load_prediction(Path(filename))
        predictions.append(pred)
        f2s.append(f2)
        print('{:.5f} {}'.format(f2, filename))
    print('Mean score: {:.5f}'.format(np.mean(f2s)))

    prediction = get_df_prediction(
        pd.concat(predictions).groupby(level=0).mean())
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


def load_prediction(path: Path) -> Tuple[pd.DataFrame, float]:
    valid_df = (pd.read_csv(path.parent / 'valid.csv', index_col=0)
                .groupby(level=0).mean())

    true_df = pd.read_csv(utils.DATA_ROOT / 'train_v2.csv', index_col=0)
    true_data = np.zeros((len(valid_df), dataset.N_CLASSES), dtype=np.uint8)
    for i, tags in enumerate(true_df.loc[valid_df.index]['tags']):
        for tag in tags.split():
            true_data[i, dataset.CLASSES.index(tag)] = 1

    threshold = 0.2
    f2_score = dataset.f2_score(true_data, valid_df.as_matrix() > threshold)
    print('{} with default threshold: {:.5f}'.format(path, f2_score))

    valid_predictions = get_df_prediction(valid_df)
    f2_score = dataset.f2_score(true_data, valid_predictions.as_matrix())

    test_df = pd.read_csv(path, index_col=0).groupby(level=0).mean()
    return test_df, f2_score


def get_df_prediction(df):
    data = [get_item_prediction(item)
            for _, item in tqdm.tqdm(list(df.iterrows()))]
    return pd.DataFrame(data=data, index=df.index, columns=df.columns)


def get_item_prediction(item):
    free_vars, value = get_free_vars(item)
    if not free_vars:
        return value
    ys = get_true_candidates(item, free_vars, value)
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


def get_free_vars(item, min_p=0.05, max_p=0.5):
    free_vars = []
    value = np.zeros(dataset.N_CLASSES)
    for i, p in enumerate(item):
        if p >= min_p:
            if p >= max_p:
                value[i] = 1
            else:
                free_vars.append([(i, 0), (i, 1)])
    return free_vars, value


def get_true_candidates(item, free_vars, value):
    ys = []
    for var_set in itertools.product(*free_vars):
        for i, v in var_set:
            value[i] = v
        if value.sum() == 0:
            continue
        # FIXME this two conditions make the score much worse
        if value[dataset.CLOUDY_ID] and value.sum() > 1:
            # cloudy + something = 0
            continue
        if sum(value[i] for i in dataset.WEATHER_IDS) > 1:
            # several weather tags
            continue
        prob = 1.0
        for v, p in zip(value, item):
            if v == 0:
                p = 1 - p
            prob *= p
        if prob > 1e-7:
            ys.append((value.copy(), prob))
    return ys


def f2_fast(y_true, y_pred):
    tp = y_true @ y_pred
    r = tp / y_true.sum()
    if r == 0:
        return 0
    p = tp / (y_pred.sum() + 1e-6)
    beta2 = 4
    return (1 + beta2) * p * r / (beta2 * p + r)


if __name__ == '__main__':
    main()
