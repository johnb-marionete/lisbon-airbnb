# -*- coding: utf-8 -*-

import sys

import click
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import math
import datetime as dt
    

def distance(lat1, long1, lat2, long2):
    
    R = 6371
    p1, l1, p2, l2 = map(np.deg2rad, [lat1, long1, lat2, long2])
        
    dp = np.radians(p2-p1)
    dl = np.radians(l2-l1)
 
    a = np.sin((p2-p1)/2.0)**2 + \
        np.cos(p1) * np.cos(p2) * np.sin((l2-l1)/2.0)**2

    return R * 2 * np.arcsin(np.sqrt(a))

def within_km(row, data):
    
    lat = row.latitude
    long = row.longitude
    rowid = row.id
        
    data['distance'] = distance(lat, long, data['latitude'].values, data['longitude'].values)
    dcop = data.loc[(data.distance < 1.5) & (data.id != rowid)]
    data.drop('distance',inplace=True, axis=1)
    return dcop.price.mean(), len(dcop)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ 
    Runs feature engineering scripts to enhance data 
    """

    logger = logging.getLogger(__name__)
    logger.info('generating features')

    # read data as pd df
    data = pd.read_csv(input_filepath, encoding="UTF-8", low_memory=False)

    columns = [
        'id', 
        'host_location', 
        'host_response_time',
        'host_response_rate', 
        'host_is_superhost', 
        'host_listings_count',
        'neighbourhood_cleansed', 
        'property_type', 
        'room_type', 
        'accommodates', 
        'bedrooms',
        'beds', 
        'price',
        'minimum_nights',
        'maximum_nights',
        'availability_30',
        'availability_60',
        'availability_90',
        'availability_365',
        'number_of_reviews',
        'review_scores_rating',
        'instant_bookable',
        'reviews_per_month',
        'latitude',
        'longitude'
    ]

    data = data[columns].copy()

    
    # calcualte neighbour values
    x = data.apply(lambda row: within_km(row, data), axis=1)
    data['nearby_average_price'] = [z[0] for z in x]
    data['n_nearby'] = [z[1] for z in x]

    print("n_nearby done")
    
    # host response time
    data.host_response_time = [1 if x == "within an hour" else 0 for x in data.host_response_time ]

    data['host_response_rate_100'] = [1 if x == "100%" else 0 for x in data.host_response_rate ]
    data.drop("host_response_rate", inplace=True, axis=1)
    data.review_scores_rating.fillna(0, inplace=True)
    data.reviews_per_month.fillna(0, inplace=True)

    data['host_in_pt'] = np.where(data.host_location.str.contains("ortugal"), 1,
                                     np.where(data.host_location.str.contains("PT"), 1,
                                     0))
    data.drop('host_location', inplace=True, axis=1)

    data['host_is_superhost'] = [1 if x == "t" else 0 for x in data.host_is_superhost]

    
    dt = pd.get_dummies(data[["neighbourhood_cleansed", "property_type", "room_type"]], prefix=['neighbourhood', 'property', 'room'], dummy_na=True)
    dt.columns = dt.columns.str.replace(' ', '_').str.lower()

    data = pd.concat([data.drop(['neighbourhood_cleansed', 'property_type', 'room_type'], axis=1), dt], axis=1)

    data['instant_bookable'] = [1 if x == "t" else 0 for x in data.instant_bookable]

    data['beds'].fillna(data['bedrooms'], inplace=True)

    data = data.loc[data.price.isna()==False]
    data.drop(["neighbourhood_nan", "property_nan", "room_nan"], inplace=True, axis=1)

    data.dropna(axis=0, inplace=True)

    data.set_index('id', inplace=True)
    
    data.to_csv(output_filepath, index=True)

    sys.exit(1)
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files                                                                                                                          
    project_dir = Path(__file__).resolve().parents[2]

    main()
