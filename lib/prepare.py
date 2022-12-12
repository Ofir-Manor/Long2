# ===== Imports:
import numpy as np
import pandas as pd
import functools
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#========================

# ===== Aux Functions:

def _get_norm_features_lists(df: pd.DataFrame) -> tuple:
    min_max_features = ['age', 'female', 'male'] + \
    ['PCR_0' + str(i) for i in [1,2,3,4,5,7,9]] + \
    ['PCR_10'] + \
    df.filter(like='symptom').columns.tolist() + \
    df.filter(like='blood').columns.tolist()
    z_score_features = df.columns.difference(min_max_features + ['risk', 'spread']).tolist()
    return (min_max_features, z_score_features)

def _blood_type_divide(df: pd.DataFrame) -> None:
    groups = {'blood_group_A' : {'A+', 'A-'},
            'blood_group_AB_or_B' : {'AB+', 'AB-', 'B+', 'B-'},
            'blood_group_O' : {'O+', 'O-'}}
    blood_loc = df.columns.get_loc("blood_type")
    for i, (group_name, group) in enumerate(groups.items()):
        df.insert(blood_loc + 1 + i, group_name, df[
            "blood_type"].isin(group).astype(int))
    df.drop('blood_type', axis=1, inplace=True)

def _symptoms_divide(df: pd.DataFrame) -> None:
    symptoms_lists = [set(e.split(";")) for e in df['symptoms'].value_counts().index]
    symptoms = functools.reduce(lambda set1, set2: set1 | set2, symptoms_lists)

    sym_loc = df.columns.get_loc("symptoms")
    df.replace('NAN', np.nan, inplace=True)
    for i, symptom in enumerate(symptoms):
        new_col = df['symptoms'].str.contains(symptom, na=False).astype(int)
        df.insert(sym_loc + i, f"symptom {symptom}", new_col)
    df.drop('symptoms', axis=1, inplace=True)

def _sex_divide(df: pd.DataFrame) -> None:
    sex_loc = df.columns.get_loc("sex")
    for i, (sex, col_name) in enumerate(zip(['F', 'M'], ["female", "male"])):
        new_col = (df['sex'] == sex).astype(int)
        df.insert(sex_loc + 1 + i, col_name, new_col)
    df.drop('sex', axis=1, inplace=True)

def _get_cords_from_loc(curr_loc: str):
    """Function to get cordinations as `(x,y)` from the string in the column.
    """
    cords_str = curr_loc.replace("'", "").strip("()").split(", ")
    return (float(cords_str[0]), float(cords_str[1]))

def _location_divide(df: pd.DataFrame) -> None:
    location_loc = df.columns.get_loc("current_location")
    for i, col_name in enumerate(['current_location_x', 'current_location_y']):
        new_col = df.current_location.apply(
            lambda s: _get_cords_from_loc(s)[i])
        df.insert(location_loc + 1 + i, col_name, new_col)
    df.drop('current_location', axis=1, inplace=True)

def _data_divide(df: pd.DataFrame) -> None:
    for fun in [
        _blood_type_divide,
        _symptoms_divide,
        _sex_divide,
        _location_divide]:
        fun(df)

#========================

# ===== Module Functions:

def prepare_data(training_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    training_copy, retuned_df = training_data.copy(), new_data.copy()
    
    for df in [training_copy, retuned_df]:
        drop_features = ['patient_id', 'pcr_date']
        # Drop unnecessary features:
        df.drop(labels=drop_features, axis=1, inplace=True)
        # Divide the data:
        _data_divide(df)

    # Normalize:
    min_max_features, z_score_features = _get_norm_features_lists(retuned_df)
    min_max_scaler, z_score_scaler = MinMaxScaler((-1,1)), StandardScaler()
    
    min_max_scaler.fit(training_copy[min_max_features])
    retuned_df[min_max_features] = min_max_scaler.transform(retuned_df[min_max_features])

    z_score_scaler.fit(training_copy[z_score_features])
    retuned_df[z_score_features] = z_score_scaler.transform(retuned_df[z_score_features])
    
    return retuned_df
