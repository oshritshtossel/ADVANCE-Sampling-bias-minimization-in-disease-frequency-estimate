import pandas as pd

def fix_corona_raw_data_with_nan(corona_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function changes the val for True or False, changes some names of columns.
    Translate nulls to the number 5
    :param corona_df: raw data frame
    :return: no null data frame with val we want
    """
    # Maybe we need to do a better job in deleting null data
    corona_df = corona_df.rename(columns={"test_indication": "reason for check"})

    replace_dict = {"0": False, "1": True, "אחר": None, "חיובי": True, "שלילי": False, "No": False, "Yes": True,
                    "Other": False, "Other": "0", "Abroad": "2", "Contact with confirmed": "1", "נקבה": "F", "זכר": "M",
                    None: "5", "NULL": "5"}
    return corona_df.replace(replace_dict).dropna()

def vectorize_featurea_corona_df(corona_df: pd.DataFrame, count_only_positive=False,with_age=False) -> pd.DataFrame:
    """
    Transform the columns of features of the data frame to one column of a vector of the feature values
    :param corona_df: original data frame
    :param count_only_positive: a bool to determine whether to vectorize the whole data or only the data of the positive tags
    :return: new data frame consists of only 3 columns:date, vectorized features and tag
    """
    # Vectorizing only the data of the positive tags:
    if count_only_positive is True:
        corona_df = corona_df[corona_df["corona_result"] == True]
    vectorize_df = corona_df.replace({False: "0", True: "1"})

    if with_age == True:
        vectorize_df['features_vector'] = vectorize_df['cough'].astype(str) + vectorize_df['fever'].astype(str) + \
                                          vectorize_df['sore_throat'].astype(str) + vectorize_df[
                                              'shortness_of_breath'].astype(str) + vectorize_df['head_ache'].astype(
            str) + \
                                          vectorize_df['gender'].astype(
                                              str) + vectorize_df['reason for check'].astype(
            str)  + vectorize_df['age_60_and_above'].astype(str)
        return vectorize_df.drop(columns=["cough", "fever", "sore_throat", "shortness_of_breath", "head_ache",
                                          "gender",
                                          "reason for check","age_60_and_above"])
    else:
        vectorize_df['features_vector'] = vectorize_df['cough'].astype(str) + vectorize_df['fever'].astype(str) + \
                                      vectorize_df['sore_throat'].astype(str) + vectorize_df[
                                          'shortness_of_breath'].astype(str) + vectorize_df['head_ache'].astype(str) + \
                                       vectorize_df['gender'].astype(
        str) + vectorize_df['reason for check'].astype(str) #vectorize_df['age_60_and_above'].astype(str)

    return vectorize_df.drop(columns=["cough", "fever", "sore_throat", "shortness_of_breath", "head_ache",
                                       "gender",
                                      "reason for check"])#"age_60_and_above"


def mul_series(s1, s2, index=None, columns=None):
    if index is None:
        index = s1.index
    if columns is None:
        columns = s2.index

    p0t_mul_qi_df = pd.DataFrame(1, index=index, columns=columns)
    return p0t_mul_qi_df.multiply(s1, axis='index') * s2
