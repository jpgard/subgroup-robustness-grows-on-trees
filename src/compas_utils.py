"""
Utilities for working with the COMPAS dataset.
"""

import pandas as pd
from datetime import datetime


def months_since(date_str, year=2013, month=1):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    return (date.year - year) * 12 + (date.month - month)


def preprocess_compas(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess COMPAS dataset.

    See https://github.com/RuntianZ/doro/blob/master/compas.py .
    """

    columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count',
               'priors_count',
               'age',
               'c_charge_degree',
               'sex', 'race', 'is_recid', 'compas_screening_date']

    df = df[['id'] + columns].drop_duplicates()
    df = df[columns]

    race_dict = {'African-American': 1, 'Caucasian': 0}
    df['race'] = df.apply(
        lambda x: race_dict[x['race']] if x['race'] in race_dict.keys() else 2,
        axis=1).astype(
        'category')

    # Screening dates are either in year 2013, or 2014.
    df['screening_year_is_2013'] = df['compas_screening_date'].apply(
        lambda x: int(datetime.strptime(x, "%Y-%m-%d").year == 2013))
    df.drop(columns=['compas_screening_date'], inplace=True)

    sex_map = {'Female': 0, 'Male': 1}
    df['sex'] = df['sex'].map(sex_map)

    c_charge_degree_map = {'F': 0, 'M': 1}
    df['c_charge_degree'] = df['c_charge_degree'].map(c_charge_degree_map)

    return df
