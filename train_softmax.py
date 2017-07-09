#!/usr/bin/env python3
import argparse
from itertools import chain
import json
from pathlib import Path
import random
import shutil
from typing import Dict, List

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
import tqdm

import augmentation
import dataset
import models
import utils


class PlanetDataset(Dataset):
    def __init__(self, paths: List[Path], train_labels, classes):
        self.images = {p.stem: utils.load_image(p) for p in tqdm.tqdm(paths)}
        self.keys = sorted(self.images)
        id_by_tags = {tags: i for i, tags in enumerate(classes)}
        self.tags = classes
        self.train_labels = {
            stem: id_by_tags[tags]
            for stem, tags in zip(train_labels['image_name'],
                                  train_labels['tags'])}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img = self.images[key]
        return augmentation.img_transform(img), self.train_labels[key]


def validation(model: nn.Module, criterion, valid_loader) -> Dict[str, float]:
    loader = (valid_loader.iterable if not isinstance(valid_loader, DataLoader)
              else valid_loader)
    tags = loader.dataset.tags
    model.eval()
    losses = []
    f2_scores, f2_bayes_scores = [], []

    f_matrix_path = Path('f_matrix.npy')
    if f_matrix_path.exists():
        with f_matrix_path.open('rb') as f:
            f_matrix = np.load(f)
    else:
        f_matrix = np.zeros((len(tags), len(tags)))
        tags_as_ids = [tag_ids_to_matrix([i], tags) for i in range(len(tags))]
        for i, i_tag_ids in enumerate(tqdm.tqdm(tags_as_ids, desc='F matrix')):
            for j, j_tag_ids in enumerate(tags_as_ids):
                if i >= j:
                    f_matrix[i, j] = f_matrix[j, i] = (
                        dataset.f2_score(i_tag_ids, j_tag_ids))
        with f_matrix.open('wb') as f:
            np.save(f, f_matrix)

    for inputs, targets in valid_loader:
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])

        probs = F.softmax(outputs).data.cpu().numpy()
        y_pred = tag_ids_to_matrix(probs.argmax(axis=1), tags)
        y_true = tag_ids_to_matrix(targets.data.cpu().numpy(), tags)
        f2_scores.append(dataset.f2_score(y_true=y_true, y_pred=y_pred))

        y_pred_bayes = tag_ids_to_matrix((probs @ f_matrix).argmax(axis=1), tags)
        f2_bayes_scores.append(
            dataset.f2_score(y_true=y_true, y_pred=y_pred_bayes))

    valid_loss = np.mean(losses)  # type: float
    valid_f2 = np.mean(f2_scores)  # type: float
    valid_f2_bayes = np.mean(f2_bayes_scores)  # type: float
    print('Valid loss: {:.4f}, F2 naive: {:.4f}, F2 bayes: {:.4f}'
          .format(valid_loss, valid_f2, valid_f2_bayes))
    return {'valid_loss': valid_loss, 'valid_f2': valid_f2,
            'valid_f2_bayes': valid_f2_bayes}


def tag_ids_to_matrix(tag_ids, tags):
    matrix = np.zeros((len(tag_ids), dataset.N_CLASSES))
    for i, tag_id in enumerate(tag_ids):
        for tag in tags[tag_id].split():
            matrix[i, dataset.CLASSES.index(tag)] = 1
    return matrix


class PredictionDataset:
    def __init__(self, paths: List[Path], n_test_aug):
        self.paths = paths
        self.n_test_aug = n_test_aug

    def __len__(self):
        return len(self.paths) * self.n_test_aug

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image = utils.load_image(path)
        return augmentation.img_transform(image), path.stem


def predict(model, paths: List[Path], out_path: Path, batch_size: int,
            n_test_aug: int):
    loader = DataLoader(
        dataset=PredictionDataset(paths, n_test_aug),
        shuffle=False,
        batch_size=batch_size,
        num_workers=1,
    )
    model.eval()
    all_outputs = []
    all_stems = []
    for inputs, stems in tqdm.tqdm(loader, desc='Predict'):
        inputs = utils.variable(inputs, volatile=True)
        assert False  # TODO
        outputs = F.sigmoid(model(inputs))
        all_outputs.append(outputs.data.cpu().numpy())
        all_stems.extend(stems)
    all_outputs = np.concatenate(all_outputs)
    df = pd.DataFrame(data=all_outputs, index=all_stems,
                      columns=dataset.CLASSES)
    df.to_csv(out_path)
    print('Saved predictions to {}'.format(out_path))


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('model')
    arg('--mode', choices=['train', 'valid', 'predict_valid', 'predict_test'],
        default='train')
    arg('--limit', type=int, help='use only N images for valid/train')
    arg('--n-test-aug', type=int, default=5,
        help='number of test time augmentations')
    utils.add_args(parser)
    args = parser.parse_args()

    train_labels = pd.read_csv(utils.DATA_ROOT / 'train_v2.csv')
    classes = sorted(train_labels['tags'].unique())

    root = Path(args.root)
    model = getattr(models, args.model)(num_classes=len(classes))
    model = utils.cuda(model)
    loss = nn.CrossEntropyLoss()

    random.seed(1)
    paths = sorted(utils.DATA_ROOT.joinpath('train-jpg').glob('*.jpg'))
    if args.limit:
        paths = random.sample(paths, args.limit)
    train_paths, valid_paths = utils.train_valid_split(args, paths)

    def make_loader(paths):
        return DataLoader(
            dataset=PlanetDataset(paths, train_labels, classes),
            shuffle=True,
            num_workers=args.workers,
            batch_size=args.batch_size,
        )

    if args.mode == 'train':
        train_loader, valid_loader = map(make_loader,
                                         [train_paths, valid_paths])
        if root.exists() and args.clean:
            shutil.rmtree(str(root))
        root.mkdir(exist_ok=True)
        root.joinpath('params.json').write_text(
            json.dumps(vars(args), indent=True, sort_keys=True))
        train_kwargs = dict(
            args=args,
            model=model,
            criterion=loss,
            train_loader=train_loader,
            valid_loader=valid_loader,
            validation=validation,
            patience=4,
        )
        if getattr(model, 'finetune', None):
            utils.train(
                init_optimizer=lambda lr: Adam(model.fresh_params(), lr=lr),
                n_epochs=1,
                **train_kwargs)
            utils.train(
                init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
                **train_kwargs)
        else:
            utils.train(
                init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
                **train_kwargs)

    elif args.mode == 'valid':
        valid_loader = make_loader(valid_paths)
        state = torch.load(str(Path(args.root) / 'model.pt'))
        model.load_state_dict(state['model'])
        validation(model, loss, tqdm.tqdm(valid_loader, desc='Validation'))

    elif args.mode == 'predict_valid':
        utils.load_best_model(model, root)
        predict(model, valid_paths, out_path=root / 'valid.csv',
                batch_size=args.batch_size,
                n_test_aug=args.n_test_aug)

    elif args.mode == 'predict_test':
        utils.load_best_model(model, root)
        test_paths = list(chain(
            (utils.DATA_ROOT / 'test-jpg').glob('*.jpg'),
            (utils.DATA_ROOT / 'test-jpg-additional').glob('*.jpg')))
        predict(model, test_paths, out_path=root / 'test.csv',
                batch_size=args.batch_size,
                n_test_aug=args.n_test_aug)


if __name__ == '__main__':
    main()
