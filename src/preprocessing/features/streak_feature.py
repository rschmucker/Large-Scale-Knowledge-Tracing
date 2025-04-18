import os
import pandas as pd

def extract_streak_feature(data_dict):
    
    df = data_dict["interaction_df"].copy()
    save_path = data_dict["dataset"]
    save_path = f"./data/{save_path}/features/streak.pkl"

    df.sort_values(by=["user_id", "timestamp"], inplace=True)

    streaks = []
    current_streak = 0
    previous_user = None

    for _, row in df.iterrows():
        if row["user_id"] != previous_user:
            current_streak = 0
            previous_user = row["user_id"]
        
        streaks.append(current_streak)

        if row["correct"] == 1:
            current_streak += 1
        else:
            current_streak = 0

    df["streak"] = pd.Series(streaks, dtype="Sparse[int]")
    
    # Reuse U_ID already added during load_preprocessed_data
    result_df = df[["U_ID", "streak"]]

    result_df = df[["U_ID", "streak"]]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result_df.to_pickle(save_path)
    
    print("✅ streak.pkl saved to:", save_path)
    print("✅ Columns in saved DataFrame:", result_df.columns.tolist())
    print("✅ First 5 rows:\n", result_df.head())

    
    
if __name__ == "__main__":
    import sys
    from src.utils.data_loader import load_interaction_df

    dataset = [arg.split("=")[1] for arg in sys.argv if arg.startswith("--dataset=")][0]
    print(f"🧠 Dataset received: {dataset}")

    data_dict = {
        "dataset": dataset,
        "interaction_df": load_interaction_df(dataset)
    }

    extract_streak_feature(data_dict)

