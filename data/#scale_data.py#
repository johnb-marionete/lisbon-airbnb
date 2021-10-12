# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('scaler_filepath', type=click.Path())
@click.option('--train', '-t', is_flag=True, help="Declare training use")
def main(input_filepath, output_filepath, scaler_filepath, train):
    """ 
    Scale data for modelling
    """
    
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data = pd.read_csv(input_filepath, encoding="UTF-8", low_memory=False)

    if train:

        logger.info(f"Training: creating new scaler {scaler_filepath}")
        
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data.drop(["price", "id"], axis=1)), columns=data.drop(["price", "id"], axis=1).columns)

        data_scaled["price"] = data["price"].copy()
        data_scaled["id"] = data["id"].copy()
        
        pickle.dump(scaler, open(scaler_filepath, 'wb'))

    else:

        logger.info(f"Inference: Scaling from {scaler_filepath}")
        
        scaler = pickle.load(open(scaler_filepath, 'rb'))

        data_scaled = pd.DataFrame(scaler.transform(data.drop(["price", "id"], axis=1)), columns=data.drop(["price", "id"], axis=1).columns)

        data_scaled['price'] = data['price'].copy()
        data_scaled["id"] = data["id"].copy()


    data_scaled.set_index("id", inplace=True)

    data_scaled.to_csv(output_filepath)
        
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
