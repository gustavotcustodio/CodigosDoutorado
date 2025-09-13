import argparse
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

def sample_random_dataset(
        df_data: DataFrame, n_samples_class: int, label_col: int,
) -> NDArray:
     
    data = df_data.values

    labels = np.unique(data[:, label_col])
    samples_by_class = []

    for c in labels:
        data_class = data[data[:, -1]==c, :]
        idx_samples = np.random.choice(
            range(len(data_class)), size=n_samples_class, replace=False
        )
        sampled_class_data = data_class[idx_samples]
        samples_by_class.append(sampled_class_data)

    return np.vstack(samples_by_class)


def save_data(
        sampled_data: NDArray, column_names: list[str], label_col: int,
        output_path: str) -> None:

    label_colname = column_names[label_col]
    df_sampled = pd.DataFrame(data=sampled_data, columns=column_names)

    df_sampled[label_colname] = df_sampled[label_colname].astype(int)
    df_sampled.to_csv(output_path, index=False)

    print(f"{output_path} saved successfully.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--dataset", type=str, required=True,
        help = "Dataset to be sampled.")
    parser.add_argument(
        "-o", "--output_path", type=str, required=True,
        help = "Output file name.")
    parser.add_argument(
        "-n", "--num_samples_by_class", type=int, required=True,
        help = "Number of samples by class to be sampled.")
    parser.add_argument(
        "-l", "--label_col", type=int, default=-1,
        help = "Column with the label.")

    args = parser.parse_args()

    df_data = pd.read_csv(args.dataset)

    sampled_data = sample_random_dataset(
        df_data, args.num_samples_by_class, args.label_col
    )

    column_names = list(df_data.columns)
    save_data(sampled_data, column_names, args.label_col, args.output_path)


if __name__ == "__main__":
    main()
