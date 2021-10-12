# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import pandas as pd
import numpy as np

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # read data as pd df
    data = pd.read_csv(input_filepath, encoding="UTF-8", low_memory=False)

    # strip url columns
    data = data[[col for col in data.columns if 'url' not in col ]]

    # drop na cols
    data = data.dropna(axis=1,how="all")

    # drop host verifications
    data.drop(labels="host_verifications",axis=1,inplace=True)

    # drop homogenous columns
    data = data[[c for c
        in list(data)
        if len(data[c].unique()) > 1]]

    # set price col as float
    cols = [
        "price" 
    ]
    data[cols] = data[cols].replace({'\$': '', ',': ''}, regex=True).astype(float)

    # drop wordy columns
    wordy_cols = [
        "name",
        "description",
        "neighborhood_overview",
        "host_name",
        "host_about"
    ]
    data.drop(wordy_cols, axis=1, inplace=True)

    # write data to file
    data.to_csv(output_filepath, index=False)
    
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
