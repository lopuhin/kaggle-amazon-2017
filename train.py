#!/usr/bin/env python3
import argparse
from itertools import chain
import json
from pathlib import Path
import shutil
from typing import Dict, List

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, CenterCrop
import tqdm

import utils
import models
import dataset
import augmentation


class PlanetDataset(Dataset):
    def __init__(self, paths: List[Path], transform, classes: List[str]):
        self.transform = transform
        self.images = {p.stem: utils.load_image(p) for p in tqdm.tqdm(paths)}
        self.keys = sorted(self.images)
        train_labels = pd.read_csv(utils.DATA_ROOT / 'train_v2.csv')
        self.classes = classes
        self.train_labels = {
            stem: [self.classes.index(t) for t in tags.split()
                   if t in self.classes]
            for stem, tags in zip(train_labels['image_name'],
                                  train_labels['tags'])}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img = self.images[key]
        x = self.transform(img)
        y = np.zeros(len(self.classes), dtype=np.float32)
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


class RandomAugDataset:
    def __init__(self, paths: List[Path], transform, n_random_aug: int):
        self.paths = paths
        self.transform = transform
        self.n_random_aug = n_random_aug

    def __len__(self):
        return len(self.paths) * self.n_random_aug

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image = utils.load_image(path)
        return self.transform(image), path.stem


class CenterCropDataset:
    def __init__(self, paths: List[Path]):
        self.paths = paths
        self.transform = Compose([
            CenterCrop(224),  # FIXME - Inception
            utils.img_transform,  # FIXME - Inception
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image = utils.load_image(path)
        return self.transform(image), path.stem


def predict(model, paths: List[Path], out_path: Path,
            transform, batch_size: int, test_aug: str, n_random_aug: int,
            classes: List[str]):
    if test_aug == 'center':
        ds = CenterCropDataset(paths)
    elif test_aug == 'random':
        ds = RandomAugDataset(paths, transform, n_random_aug)
    else:
        raise ValueError('augmentation "{}" not supported'.format(test_aug))

    loader = DataLoader(
        dataset=ds,
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
                      columns=classes)
    df = utils.mean_df(df)
    df.to_hdf(out_path, 'prob', index_label='image_name')
    print('Saved predictions to {}'.format(out_path))


def f2_loss(outputs, targets):
    outputs = F.sigmoid(outputs)
    tp = (outputs * targets).sum(dim=1)
    r = tp / targets.sum(dim=1)
    p = tp / (outputs.sum(dim=1) + 1e-5)
    beta2 = 4
    f2 = (1 + beta2) * p * r / (beta2 * p + r)
    return 1 - f2.mean()


def load_fold(fold: int):
    fold_root = utils.DATA_ROOT / 'fold{}'.format(fold)
    valid_paths = pd.read_csv(fold_root / 'val.csv')['path']
    train_paths = pd.read_csv(fold_root / 'train.csv')['path']
    return [[Path(x) for x in paths] for paths in [train_paths, valid_paths]]


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('model')
    arg('--mode', choices=['train', 'valid', 'predict_valid', 'predict_test'],
        default='train')
    arg('--limit', type=int, help='use only N images for valid/train')
    arg('--f2-loss', action='store_true')
    arg('--scale-aug', action='store_true')
    arg('--stratified', action='store_true')
    arg('--test-aug', choices=['center', 'random', '12crops', 'scaled'])
    arg('--n-random-aug', type=int, default=8,
        help='number of random test time augmentations')
    arg('--classes', help='dash separated, e.g. road-water')
    utils.add_args(parser)
    args = parser.parse_args()

    root = Path(args.root)
    if args.classes:
        classes = sorted(args.classes.split('-'))
    else:
        classes = dataset.CLASSES
    model = getattr(models, args.model)(num_classes=len(classes))
    model = utils.cuda(model)
    loss = f2_loss if args.f2_loss else nn.MultiLabelSoftMarginLoss()

    train_paths, valid_paths = load_fold(args.fold)
    if args.limit:
        train_paths = train_paths[:args.limit]
        valid_paths = train_paths[:args.limit // 10]

    aug_transform = (augmentation.with_scale_transform if args.scale_aug else
                     augmentation.default_transform)
    transform = Compose([
        aug_transform,
        utils.img_transform_inception if 'inception' in args.model
        else utils.img_transform,
    ])

    def make_loader(paths):
        return DataLoader(
            dataset=PlanetDataset(paths, transform, classes),
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

    elif args.mode.startswith('predict'):
        utils.load_best_model(model, root)
        predict_kwargs = dict(
            classes=classes,
            batch_size=args.batch_size,
            transform=transform,
            test_aug=args.test_aug,
            n_random_aug=args.n_random_aug,
        )
        if args.mode == 'predict_valid':
            predict(model, valid_paths,
                    out_path=root / 'val_{}.h5'.format(args.test_aug),
                    **predict_kwargs)
        elif args.mode == 'predict_test':
            test_paths = list(chain(
                (utils.DATA_ROOT / 'test-jpg').glob('*.jpg'),
                (utils.DATA_ROOT / 'test-jpg-additional').glob('*.jpg')))
            predict(model, test_paths,
                    out_path=root / 'test_{}.h5'.format(args.test_aug),
                    **predict_kwargs)


if __name__ == '__main__':
    main()
