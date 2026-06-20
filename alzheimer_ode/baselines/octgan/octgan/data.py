import json
import logging
import os

import numpy as np
import pandas as pd

from octgan.constants import CATEGORICAL, ORDINAL

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(filename, loader):
    local_path = os.path.join(DATA_PATH, filename)
    
    if loader == np.load:
        return loader(local_path, allow_pickle=True)
    return loader(local_path)


def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)

    return categorical_columns, ordinal_columns


def load_dataset(name, benchmark=False):
    LOGGER.info('Loading dataset %s', name)
    data = _load_file(name + '.npz', np.load)
    meta = _load_file(name + '.json', _load_json)

    categorical_columns, ordinal_columns = _get_columns(meta)

    train = data['train']
    test = data['test']
    if benchmark:
        return train, test, meta, categorical_columns, ordinal_columns

    return train, categorical_columns, ordinal_columns


def load_atndataset(name, benchmark=False):
    LOGGER.info('Loading dataset %s', name)
    split = 'train'
    train = pd.read_csv(f'data/{name}/{split}.csv')
    split = 'test'
    test = pd.read_csv(f'data/{name}/{split}.csv')
    discrete_columns = [c for c in train.columns.tolist() if 'cat' in c]

    if benchmark:
        return train, test, discrete_columns

    return train, discrete_columns
