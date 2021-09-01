import os
import pandas as pd
from config.constants import DATASET_PATH
from src.preprocessing.features.feature_util import phi_inv

N_AVG = 5
AVG_CORRECT = {
    "squirrel": 0.685249091469651,
    "ednet_kt3": 0.661876189869379,
    "eedi": 0.6429725856250825,
    "junyi_15": 0.8298862191292307,
}


###############################################################################
# Average user correctness
###############################################################################

def user_avg_correct(data_dict):
    """Create a dataframe containining containing avg correct per user so far

    Arguments:
        data_dict (dict): contains all relevant information
    Output:
        user_avg_correct_df (pandas DataFrame): df user avg correct over time
    """
    # Ensure that required features exist
    feature_path = DATASET_PATH[data_dict["dataset"]] + "features/"
    for f in ["tcA", "tcW"]:
        from src.preprocessing.extract_features import extract_features
        if not os.path.isfile(feature_path + f + ".pkl"):
            extract_features([f], data_dict, recompute=False)
    n = data_dict.get('n_avg', N_AVG)

    # Load lower-order features
    print("\nLoading tcA")
    tcA = pd.read_pickle(feature_path + "tcA.pkl")
    tcA = phi_inv(tcA["tcA"].values)
    print("Loading tcW")
    tcW = pd.read_pickle(feature_path + "tcW.pkl")
    tcW = phi_inv(tcW["tcW"].values)

    data = data_dict["interaction_df"]
    columns = [c for c in list(data.columns) if c not in ['U_ID']]
    user_avg_correct_df = data.drop(columns=columns, inplace=False)

    # Smoothed average correctness over time
    avg = AVG_CORRECT[data_dict["dataset"]]
    user_avg_correct_df["user_avg_correct"] = ((tcW + (avg * n)) / (tcA + n))
    return user_avg_correct_df
