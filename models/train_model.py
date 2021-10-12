# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

import yaml

import sys

from pydoc import locate

from joblib import dump

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('model_config', type=click.Path())
@click.argument('output_metrics', type=click.Path())
#@click.option('--train', '-t', is_flag=True, help="Declare training use")
def main(input_filepath, model_filepath, model_config, output_metrics):
    """ 
    train model
    """
    
    logger = logging.getLogger(__name__)
    logger.info('train model')

    # Load model config parameters
    
    with open(model_config) as f:
        model_params = yaml.load(f, Loader=yaml.FullLoader)

        print(model_params)

    clf = locate(f'{model_params["type"]}.{model_params["estimator"]}')(**model_params["parameters"])

    data = pd.read_csv(input_filepath, encoding="UTF-8", low_memory=False, index_col="id")

    print([c for c in data.columns])
    
    y = np.log(data['price'])

    X = data.drop('price', axis=1).copy()

    clf.fit(X, y)

    scores = cross_validate(clf, X, y, cv=5, scoring=('neg_mean_squared_error', 'r2'))

    print(scores)
    
    dump(clf, model_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
