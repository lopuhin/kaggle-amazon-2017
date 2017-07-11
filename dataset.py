import numpy as np
from sklearn.metrics import fbeta_score


CLASSES = [
    'agriculture',
    'artisinal_mine',
    'bare_ground',
    'blooming',
    'blow_down',
    'clear',
    'cloudy',
    'conventional_mine',
    'cultivation',
    'habitation',
    'haze',
    'partly_cloudy',
    'primary',
    'road',
    'selective_logging',
    'slash_burn',
    'water',
]
N_CLASSES = len(CLASSES)

RARE_CLASSES = [
    'blow_down',          #  98 (0.24%)
    'conventional_mine',  # 100 (0.25%)
    'slash_burn',         # 209 (0.52%)
    'blooming',           # 332 (0.82%)
    'artisinal_mine',     # 339 (0.84%)
    'selective_logging',  # 340 (0.84%)
]

WEATHER_CLASSES = {'clear', 'cloudy', 'haze', 'partly_cloudy'}
WEATHER_IDS = {i for i, cls in enumerate(CLASSES) if cls in WEATHER_CLASSES}
CLOUDY_ID = CLASSES.index('cloudy')


def f2_score(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-7) -> float:
    # same as fbeta_score(y_true, y_pred, beta=2, average='samples')
    # but faster
    tp = (y_true * y_pred).sum(axis=1)
    r = tp / y_true.sum(axis=1)
    p = tp / (y_pred.sum(axis=1) + eps)
    beta2 = 4
    f2 = (1 + beta2) * p * r / (beta2 * p + r + eps)
    return f2.mean()
