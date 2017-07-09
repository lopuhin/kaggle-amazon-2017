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
from torchvision.transforms import Compose
import tqdm

import utils
import models
import dataset
import augmentation


class PlanetDataset(Dataset):
    def __init__(self, paths: List[Path], transform):
        self.transform = transform
        self.images = {p.stem: utils.load_image(p) for p in tqdm.tqdm(paths)}
        self.keys = sorted(self.images)
        train_labels = pd.read_csv(utils.DATA_ROOT / 'train_v2.csv')
        self.train_labels = {
            stem: [dataset.CLASSES.index(t) for t in tags.split()]
            for stem, tags in zip(train_labels['image_name'],
                                  train_labels['tags'])}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img = self.images[key]
        x = self.transform(img)
        y = np.zeros(dataset.N_CLASSES, dtype=np.float32)
        for cls in self.train_labels[key]:
            y[cls] = True
        return x, torch.from_numpy(y)


def validation(model: nn.Module, criterion, valid_loader) -> Dict[str, float]:
    model.eval()
    losses = []
    f2_scores = []
    for inputs, targets in valid_loader:
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        f2_scores.append(dataset.f2_score(
            y_true=targets.data.cpu().numpy(),
            y_pred=F.sigmoid(outputs).data.cpu().numpy() > 0.2))
    valid_loss = np.mean(losses)  # type: float
    valid_f2 = np.mean(f2_scores)  # type: float
    print('Valid loss: {:.4f}, F2: {:.4f}'.format(valid_loss, valid_f2))
    return {'valid_loss': valid_loss, 'valid_f2': valid_f2}


class PredictionDataset:
    def __init__(self, paths: List[Path], transform, n_test_aug: int):
        self.paths = paths
        self.transform = transform
        self.n_test_aug = n_test_aug

    def __len__(self):
        return len(self.paths) * self.n_test_aug

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image = utils.load_image(path)
        return self.transform(image), path.stem


def predict(model, paths: List[Path], out_path: Path,
            transform, batch_size: int, n_test_aug: int):
    loader = DataLoader(
        dataset=PredictionDataset(paths, transform, n_test_aug),
        shuffle=False,
        batch_size=batch_size,
        num_workers=1,
    )
    model.eval()
    all_outputs = []
    all_stems = []
    for inputs, stems in tqdm.tqdm(loader, desc='Predict'):
        inputs = utils.variable(inputs, volatile=True)
        outputs = F.sigmoid(model(inputs))
        all_outputs.append(outputs.data.cpu().numpy())
        all_stems.extend(stems)
    all_outputs = np.concatenate(all_outputs)
    df = pd.DataFrame(data=all_outputs, index=all_stems,
                      columns=dataset.CLASSES)
    df.to_csv(out_path)
    print('Saved predictions to {}'.format(out_path))


def f2_loss(outputs, targets):
    outputs = F.sigmoid(outputs)
    tp = (outputs * targets).sum(dim=1)
    r = tp / targets.sum(dim=1)
    p = tp / (outputs.sum(dim=1) + 1e-5)
    beta2 = 4
    f2 = (1 + beta2) * p * r / (beta2 * p + r)
    return 1 - f2.mean()


def stratified_split(args, paths: List[Path]):
    train_labels = pd.read_csv(utils.DATA_ROOT / 'train_v2.csv')
    sort_keys = {stem: [c in tags.split() for c in dataset.RARE_CLASSES]
                 for stem, tags in zip(train_labels['image_name'],
                                       train_labels['tags'])}
    paths = list(paths)
    random.seed(1)
    random.shuffle(paths)
    train_paths, valid_paths = [], []
    for i, p in enumerate(sorted(paths, key=lambda p: sort_keys[p.stem])):
        if i % args.n_folds == args.fold - 1:
            valid_paths.append(p)
        else:
            train_paths.append(p)
    return train_paths, valid_paths


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('model')
    arg('--mode', choices=['train', 'valid', 'predict_valid', 'predict_test'],
        default='train')
    arg('--limit', type=int, help='use only N images for valid/train')
    arg('--n-test-aug', type=int, default=5,
        help='number of test time augmentations')
    arg('--f2-loss', action='store_true')
    arg('--scale-aug', action='store_true')
    arg('--stratified', action='store_true')
    utils.add_args(parser)
    args = parser.parse_args()

    root = Path(args.root)
    model = getattr(models, args.model)(num_classes=dataset.N_CLASSES)
    model = utils.cuda(model)
    loss = f2_loss if args.f2_loss else nn.MultiLabelSoftMarginLoss()

    random.seed(1)
    paths = sorted(utils.DATA_ROOT.joinpath('train-jpg').glob('*.jpg'))
    if args.limit:
        paths = random.sample(paths, args.limit)
    split_fn = stratified_split if args.stratified else utils.train_valid_split
    train_paths, valid_paths = split_fn(args, paths)

    aug_transform = (augmentation.with_scale_transform if args.scale_aug else
                     augmentation.default_transform)
    transform = Compose([
        aug_transform,
        utils.img_transform_inception if 'inception' in args.model
        else utils.img_transform,
    ])

    def make_loader(paths):
        return DataLoader(
            dataset=PlanetDataset(paths, transform),
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
                transform=transform,
                batch_size=args.batch_size,
                n_test_aug=args.n_test_aug)

    elif args.mode == 'predict_test':
        utils.load_best_model(model, root)
        test_paths = list(chain(
            (utils.DATA_ROOT / 'test-jpg').glob('*.jpg'),
            (utils.DATA_ROOT / 'test-jpg-additional').glob('*.jpg')))
        predict(model, test_paths, out_path=root / 'test.csv',
                transform=transform,
                batch_size=args.batch_size,
                n_test_aug=args.n_test_aug)


if __name__ == '__main__':
    main()
