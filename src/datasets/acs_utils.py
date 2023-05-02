import re
from textwrap import wrap
from typing import List

import folktables
import frozendict
import numpy as np
import pandas as pd

from src.acs_feature_mappings import get_feature_mapping
from src.datasets import SENS_RACE, SENS_SEX

# copied from (non-importable) location in folktables.load_acs
_STATE_CODES = {'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
                'CO': '08', 'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13',
                'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19',
                'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24',
                'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29',
                'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
                'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39',
                'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45',
                'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50',
                'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55', 'WY': '56',
                'PR': '72'}

# Maps 2-digit numeric strings ('01') to
# human-readable 2-letter state names ('WA')
ST_CODE_TO_STATE = {v: k for k, v in _STATE_CODES.items()}

ACS_IPR_FEATURES = ['AGEP', 'SCHL', 'MAR', 'DIS', 'ESP', 'MIG',
                    'CIT', 'MIL', 'ANC', 'NATIVITY', 'RELP', 'DEAR',
                    'DEYE', 'DREM', 'GCL', 'ESR', 'OCCP',
                    'WKHP']
ACS_TT_FEATURES = ['AGEP', 'SCHL', 'MAR', 'DIS', 'ESP', 'MIG',
                   'RELP', 'PUMA', 'CIT', 'OCCP', 'JWTR',
                   'POWPUMA', 'POVPIP']

# Default folktables features for this prediction task; see
# https://github.com/zykls/folktables/blob/12358d1645d09904b4b05e8459042e39a0d50382/folktables/acs.py#L167
ACS_PUBCOV_FEATURES = ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP',
                       'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR',
                       'DEYE', 'DREM', 'PINCP', 'ESR', 'ST',
                       'FER', 'RAC1P', ]

ACS_INCOME_FEATURES = [
'AGEP',  # Age
'CIT',  # Citizenship
'COW',  # Class of worker
# This feature is not present in source data.
# 'DIVISION',  # Division code based on 2010 Census definitions
'ENG',  # Ability to speak English
'FER',  # Gave birth to child within the past 12 months
'HINS1',  # Insurance through a current or former employer or union
'HINS2',  # Insurance purchased directly from an insurance company
'HINS3',
# Medicare, for people 65 and older, or people with certain disabilities
'HINS4',
# Medicaid, Medical Assistance, or any kind of government-assistance
# plan for those with low incomes or a disability
'MAR',  # Marital status
'NWLA',
# On layoff from work (Unedited-See "Employment Status Recode" (ESR))
'NWLK',
# Looking for work (Unedited-See "Employment Status Recode" (ESR))
'OCCP',  # Occupation recode
'POBP',  # Place of birth
'RELP',  # Relationship
'SCHL',  # Educational attainment
'ST',  # State; should be handled as grouping var.
'WKHP',  # Usual hours worked per week past 12 months
'WKW',  # Weeks worked during past 12 months
'WRK',  # Worked last week
]

def acs_travel_time_filter(data):
    """
    Modified version of folktables.acs.travel_time_filter to drop na-valued targets.
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['PWGTP'] >= 1]
    df = df[df['ESR'] == 1]
    df = df[pd.notna(df['JWMNP'])]
    return df


def acs_ipr_filter(data):
    """Drops observations with null targets."""
    df = data
    return df[pd.notna(df['POVPIP'])]


def default_acs_group_transform(x):
    # 'White alone' vs. all other categories (RAC1P) or
    # 'Male' vs. Female (SEX)
    # Note that *privileged* group is coded as 1, by default.
    return x == 1


def default_acs_postprocess(x):
    return np.nan_to_num(x, -1)


def income_cls_target_transform(y, threshold):
    """Binarization target transform for income."""
    return y > threshold

def pubcov_target_transform(y):
    """Default Public Coverage target transform from folktables."""
    return y == 1

ACS_TASK_CONFIGS = frozendict.frozendict({
    'income': {
        'features_to_use': ACS_INCOME_FEATURES,
        'group_transform': default_acs_group_transform,
        'postprocess': default_acs_postprocess,
        'preprocess': folktables.acs.adult_filter,
        'target': 'PINCP',
        'target_transform': income_cls_target_transform,
        'threshold': 56000,
    },
    'pubcov': {
        'features_to_use': ACS_PUBCOV_FEATURES,
        'group_transform': default_acs_group_transform,
        'postprocess': default_acs_postprocess,
        'preprocess': folktables.acs.public_coverage_filter,
        'target': 'PUBCOV',
        'target_transform': pubcov_target_transform,
        'threshold': None,
    },
    'tt': {
        'features_to_use': ACS_TT_FEATURES,
        'group_transform': default_acs_group_transform,
        'postprocess': default_acs_postprocess,
        'preprocess': acs_travel_time_filter,
        'target': 'JWMNP',
        'target_transform': None,
        'threshold': None,  # should raise AssertionError if called.
    },
    'ipr': {
        'features_to_use': ACS_IPR_FEATURES,
        'group_transform': default_acs_group_transform,
        'postprocess': default_acs_postprocess,
        'preprocess': acs_ipr_filter,
        'target': 'POVPIP',
        'target_transform': None,
        'threshold': None  # should raise AssertionError if called.
    },
})


def acs_numeric_to_categorical(df, feature_mapping):
    """Convert a subset of features from numeric to categorical format.

    Note that this only maps a known set of features used in this work;
    there are likely many additional categorical features treated as numeric
    that could be returned by folktables!
    """
    for feature, mapping in feature_mapping.items():
        if feature in df.columns:
            assert pd.isnull(
                df[feature]).values.sum() == 0, "nan values in input"

            mapped_setdiff = set(df[feature].unique().tolist()) - set(
                list(mapping.keys()))
            assert not mapped_setdiff, "missing keys {} from mapping {}".format(
                list(mapped_setdiff), list(mapping.keys()))
            for x in df[feature].unique():
                try:
                    assert x in list(mapping.keys())
                except AssertionError:
                    raise ValueError(f"features {feature} value {x} not in "
                                     f"mapping keys {list(mapping.keys())}")
            if feature in df.columns:
                df[feature] = pd.Categorical(
                    df[feature].map(mapping),
                    categories=list(set(mapping.values())))
            assert pd.isnull(df[feature]).values.sum() == 0, \
                "nan values in output; check for non-mapped input values."
    return df


def acs_data_to_df(
        features: np.ndarray, label: np.ndarray,
        feature_names: list,
        feature_mapping: dict) -> pd.DataFrame:
    """
    Build a DataFrame from the result of folktables.BasicProblem.df_to_numpy().
    """
    ary = np.concatenate((features, label.reshape(-1, 1),), axis=1)
    df = pd.DataFrame(ary, columns=feature_names + ['target'])
    df = acs_numeric_to_categorical(df, feature_mapping=feature_mapping)
    return df


def extract_state_from_acs_name(name: str):
    states_str = re.search(".*-state(\\w+)$", name).group(1)
    states = wrap(states_str, 2)
    return states


def extract_sens_feature_from_acs_name(name: str):
    if "sex" in name.lower():
        return SENS_TO_ACS_FEATURE_MAPPING[SENS_SEX]
    elif "race" in name.lower():
        return SENS_TO_ACS_FEATURE_MAPPING[SENS_RACE]
    else:
        raise ValueError


def add_non_sensitive_features(
        features_to_use: List[str], sens: str) -> List[str]:
    for sens_attr_name, acs_feature_name in SENS_TO_ACS_FEATURE_MAPPING.items():
        if \
                (sens != sens_attr_name) \
                        and (sens != acs_feature_name) \
                        and acs_feature_name not in features_to_use:
            features_to_use.append(acs_feature_name)
    return features_to_use


def get_acs_data_source(year, root_dir='datasets/acs'):
    return folktables.ACSDataSource(survey_year=str(year),
                                    horizon='1-Year',
                                    survey='person',
                                    root_dir=root_dir)


# Mapping of sensitive feature string codes to ACS feature names

SENS_TO_ACS_FEATURE_MAPPING = {SENS_RACE: "RAC1P", SENS_SEX: "SEX"}

# Default mapping of categorical features
DEFAULT_ACS_FEATURE_MAPPING = get_feature_mapping()
