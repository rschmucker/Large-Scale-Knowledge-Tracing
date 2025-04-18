import os
import pandas as pd
import numpy as np

def extract_lag_time_cat_feature(data_dict):
    df = data_dict["interaction_df"].copy()
    df.sort_values(by=["user_id", "timestamp"], inplace=True)
    
    # Compute lag time
    df["lag_time"] = df.groupby("user_id")["timestamp"].diff().fillna(0)

    # Bin into categories: e.g. 0 = short, 1 = medium, 2 = long
    df["lag_time_cat"] = pd.cut(
        df["lag_time"],
        bins=[-1, 10, 60, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    # Set U_ID and save
    df["U_ID"] = np.arange(1, len(df) + 1)
    result_df = df[["U_ID"]]
    result_df["lag_time_cat"] = pd.arrays.SparseArray(df["lag_time_cat"])

    
    save_path = f"./data/{data_dict['dataset']}/features/lag_time_cat.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result_df.to_pickle(save_path)

    print("✅ lag_time_cat.pkl saved to:", save_path)
    print("✅ Preview:\n", result_df.head())

if __name__ == "__main__":
    import sys
    from src.utils.data_loader import load_preprocessed_data

    dataset = sys.argv[sys.argv.index("--dataset") + 1]
    print(f"🧠 Dataset received: {dataset}")

    data_dict = load_preprocessed_data(dataset)
    extract_lag_time_cat_feature(data_dict)
