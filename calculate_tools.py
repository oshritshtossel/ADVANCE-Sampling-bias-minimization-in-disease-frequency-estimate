from itertools import product

from pandas import Series, DataFrame
import pandas as pd
import numpy as np


def calculate_prob_of_features(N: Series) -> Series:
    """
    The function calculates the probability of a specific set of features in the data.
    :param N: number of checks per day and per vector of features.
    :return: series of features and the probability of each feature in the data.
    """
    nominator = N.unstack().fillna(0).sum()
    denominator = sum(N)
    return nominator / denominator


def calculate_prob_of_features_given_day(N: Series):
    N_df = N.unstack().fillna(0)
    return N_df.T / (N_df.T.sum())


def positive_given_features(all_features, nit, kit):
    """
    The function
    :param all_features:
    :param nit:
    :param kit:
    :return:
    """
    nit = nit.reorder_levels((1, 0))
    kit = kit.reorder_levels((1, 0))

    qi = pd.Series(dtype=float)
    for feature in all_features:
        try:
            qy = kit[feature].sum()
            qy = qy / nit[feature].sum()
        except KeyError:
            qy = 0
        qi[feature] = qy
    return qi


def append_non_duplicates(a, b, col=None):
    """
    adds a to b according to the index col, with prior to a
    :param a: series 1
    :param b: series 2
    :param col: no need in series, is needed in data frame only
    :return: a union b
    """
    if ((a is not None and type(a) is not pd.core.series.Series) or (
            b is not None and type(b) is not pd.core.series.Series)):
        raise ValueError('a and b must be of type pandas.core.frame.DataFrame.')
    appended_df = a.append(b)
    swapped_index_and_value_df_no_dups = pd.Series(appended_df.index.values, index=appended_df.values).drop_duplicates()
    # Swap the index and values again to return the original form
    return_seires = pd.Series(swapped_index_and_value_df_no_dups.index.values,
                              index=swapped_index_and_value_df_no_dups.values)

    if type(return_seires.index[0]) == tuple:
        return_seires.index = pd.MultiIndex.from_tuples(return_seires.index)

    return return_seires


def dppt(Z, N, X) -> list:
    """
    calcuates the probability of positive to corona given a date for all the dates
    The function calculates the probability according to "nuschat hahistabrut hashlema"
    :param Z: pd.dataframe of probabilities of being positive to corona given a date and vector of features
    :return:
    """
    return [np.average(Z[x],
                       weights=append_non_duplicates(N[x], Z[x]).sort_index()) for x in X]


def calculate_prob_of_positive_to_covid_per_day(roots, qi, prob_of_features) -> Series:
    """
    The function calculates the probability of being positive to corona on a speific day
    :param p0t_roots: p0t roots results from the model estimation
    :param qi: qi results from the model estimation
    :param prob_of_features: series of probabilities of features from the data.
    :return:
    """
    p0t_mul_qi, roots = calculate_p0t_mul_qi(roots, qi)
    return ((1 - np.exp((-1) * p0t_mul_qi)).mul(prob_of_features, axis='index')).sum()


def calculate_p0t_mul_qi(roots, qi):
    # calculate the mul of pot and qi
    qi_dataframe = qi.shift(axis=1)
    qi_dataframe[qi_dataframe.columns[0]] = qi_dataframe[qi_dataframe.columns[1]]
    p0t_mul_qi_df = qi_dataframe * roots
    return p0t_mul_qi_df, roots
