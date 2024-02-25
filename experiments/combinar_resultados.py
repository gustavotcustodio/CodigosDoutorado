import os
import numpy as np
import pandas as pd

relevant_files = [f for f in os.listdir(".") if ".csv" in f]

def get_files_from_dataset(keyword):
    dataset_files = [fname for fname in relevant_files if keyword in fname]
    return sorted(dataset_files)

def convert_csv_to_tex_row(filename):
    df = pd.read_csv(filename)
    df = df.round(3)
    df = df.drop(columns=["run","base_classifiers"])

    dataset_name = filename.split("_")[1]
    row = [dataset_name]

    for col in df.columns:
        if np.isnan(df[col][0]):
            row.append("- ")
        else:
            row.append("$%.3f \\pm %.3f$ " % (df[col][7], df[col][8]))

    return " & ".join(row) + "\\\\"

def create_table(file_group):
    table = []

    for filename in file_group:
        tex_row = convert_csv_to_tex_row(filename)
        table.append(tex_row)
    return "\n".join(table)

def save_tex_table(final_table, dataset_name):
    filename = dataset_name + ".tex"
    with open(filename, "w") as f:
        f.write(final_table)
    print(filename + " salvo com sucesso.")

if __name__ == "__main__":
    dataset_names = ["Water", "Credit Score", "Cancer", "Wine"]

    for dataset_name in dataset_names:
        filelist = get_files_from_dataset(dataset_name)
        final_table = create_table(filelist)
        save_tex_table(final_table, dataset_name)
