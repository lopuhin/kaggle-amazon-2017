#!/usr/bin/env python3
import argparse
from pathlib import Path
import pickle
from multiprocessing.pool import ThreadPool

from PIL import Image
import numpy as np
import tqdm

import utils


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('output', help='path to save matches to')
    arg('--edge-width', type=int, default=2)
    arg('--limit', type=int, help='get matches only for first N images')
    args = parser.parse_args()

    vectors_cache_path = Path('edge_vectors_{}.npz'.format(args.edge_width))
    if vectors_cache_path.exists():
        with vectors_cache_path.open('rb') as f:
            vectors = dict(np.load(f))
    else:
        paths = sorted(utils.DATA_ROOT.glob('**/*.jpg'))
        vectors = {'top': [], 'bottom': [], 'left': [], 'right': []}
        for path in tqdm.tqdm(paths, desc='Edge vectors'):
            with Image.open(str(path)) as im:
                s = 256
                e = args.edge_width
                vectors['top'].append(crop_v(im, 0, 0, s, e))
                vectors['bottom'].append(crop_v(im, 0, s - e, s, s))
                vectors['left'].append(crop_v(im, 0, 0, e, s))
                vectors['right'].append(crop_v(im, s - e, 0, s, s))
        vectors = {k: np.array(v) for k, v in vectors.items()}
        vectors['paths'] = np.array(list(map(str, paths)))
        with vectors_cache_path.open('wb') as f:
            np.savez(f, **vectors)

    def get_all_matches(arg):
        idx, path = arg
        right_v = vectors['right'][idx]
        bottom_v = vectors['bottom'][idx]
        return (path,
                best_matches(vectors['left'], right_v),
                best_matches(vectors['top'], bottom_v))

    paths = list(vectors['paths'])
    searched_paths = paths[:args.limit] if args.limit else paths
    matches = {'paths': paths, 'bottom': {}, 'right': {}}
    with ThreadPool(processes=24) as pool:
        for p, left_matches, top_matches in tqdm.tqdm(
                pool.imap(get_all_matches, enumerate(searched_paths)),
                desc='Matching', total=len(searched_paths)):
            matches['bottom'][p] = top_matches
            matches['right'][p] = left_matches

    with open(args.output, 'wb') as f:
        pickle.dump(matches, f)


def best_matches(vectors, v, top=5):
    l2 = ((vectors - v) ** 2).mean(axis=1)
    return [(l2[idx], idx) for idx in l2.argsort()[:top]]


def crop_v(im, left, upper, right, lower):
    vec = np.array(im.crop((left, upper, right, lower)).convert('RGB'),
                   dtype=np.float32)
    axis = 0 if vec.shape[0] < 20 else 1
    return vec.mean(axis=axis).ravel()


if __name__ == '__main__':
    main()