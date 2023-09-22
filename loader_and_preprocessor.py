import pandas as pd

def read_dataset(dataset_name: str) -> pd.DataFrame:
    df_potability = pd.read_csv(dataset_name)
    df_potability.fillna(df_potability.mean(), inplace=True)
    df_potability = (df_potability - df_potability.min()) / (
                     df_potability.max() - df_potability.min())
    return df_potability
