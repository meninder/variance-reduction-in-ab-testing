import numpy as np
import pandas as pd
from scipy.stats import t


def population_variance(df):
    return df['target'].var()


def population_variance_stratify(df):
    dows = df.dow.unique()
    total_rows = len(df)
    weights = [df[df['dow'] == dow]['dow'].count() / total_rows for dow in dows]
    return sum([w_ * df[df['dow'] == dow]['target'].var() for w_, dow in zip(weights, dows)])


def population_variance_cuped(df, cuped_variable='x'):
    # Calculate the cuped
    theta = df.cov()[cuped_variable]['target'] / df.cov()[cuped_variable][cuped_variable]
    df['target_cuped'] = df.target - theta * df.x

    return df['target_cuped'].var()


def mean_variance_no_reduction(df):
    return population_variance(df)/len(df)


def mean_variance_stratification(df):
    return population_variance_stratify(df)/len(df)


def mean_variance_cuped(df):
    return population_variance_cuped(df)/len(df)


def t_statistic(df_control_: pd.DataFrame, df_treatment_: pd.DataFrame, var_type='total'):

    numerator = df_control_.target.mean() - df_treatment_.target.mean()
    if var_type == 'total':
        denominator1 = np.sqrt(mean_variance_no_reduction(df_control_))
        denominator2 = np.sqrt(mean_variance_no_reduction(df_treatment_))
    elif var_type == 'stratify':
        denominator1 = np.sqrt(mean_variance_stratification(df_control_))
        denominator2 = np.sqrt(mean_variance_stratification(df_treatment_))
    elif var_type == 'cuped':
        denominator1 = np.sqrt(mean_variance_cuped(df_control_))
        denominator2 = np.sqrt(mean_variance_cuped(df_treatment_))

    denominator = denominator1 + denominator2

    return numerator/denominator


def p_value(ratio, N):
    return t.sf(np.abs(ratio), N)*2


def print_stats(df: pd.DataFrame, txt: str):
    print(f"{txt}")
    print(f"Rows: {len(df):.0f}")
    print(f"Mean: {df.target.mean():.6f}")
    print(f"Variance: {population_variance(df):.6f}")
    print(f"Variance stratified: {population_variance_stratify(df):.6f}")
    if 'x' in df.columns:
        print(f"Variance CUPED: {population_variance_cuped(df):.6f}")
    print("-"*50)


def print_expected_n(df_control_: pd.DataFrame, df_treatment_: pd.DataFrame, delta: float):
    """
    print the expected number of rows to detect a difference
    """
    print(f"Number of treatment rows: {len(df_treatment_):.0f}")
    print(f"Number of rows needed (about): {16*population_variance(df_control_) / delta**2:.0f}")
    print("-"*50)
