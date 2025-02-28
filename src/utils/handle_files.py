import pandas as pd
from json import load
from os import path, listdir
from typing import List, Union


def read_json(json_path: str):
    """
    Reads a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        Any: Data from the JSON file.
    """
    try:
        with open(json_path) as inp:
            json_data = load(inp)
        return json_data
    except Exception as error:
        raise Exception(f"Can't read json file...\n{error}")


def write_table(table_name: str, data: Union[List[List[Union[str, int]]],
                                             pd.DataFrame]) -> None:

    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data, columns=["locus", "type", "product",
                                         "sequence", "count"])
    df.to_csv(table_name, sep="\t", index=False)


def read_tsv(table_file: str) -> pd.DataFrame:
    df = pd.read_csv(table_file, sep="\t")
    return df


def get_most_recent_folder(folder_path: str) -> str:
    """
    Gets the folder path of the most recent folder inside a folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        str: Path of the most recent folder.
    """
    folders = [path.join(folder_path, folder) for folder in
               listdir(folder_path) if path.isdir(
                   path.join(folder_path, folder))]
    if not folders:
        return ""

    most_recent_file_path = max(folders, key=path.getctime)
    return most_recent_file_path


def balance_samples(complete_df: pd.DataFrame, current_df: pd.DataFrame,
                    total_samples: int, complete_label_column: str,
                    current_label_column: str) -> pd.DataFrame:
    """
    Balances a metadata table by adding new samples while maintaining
    class balance and ensuring the total number of samples does not exceed
    the specified limit.

    Args:
        complete_df (pd.DataFrame): Dataframe containing the fixed set of
        samples.
        current_df (pd.DataFrame): Dataframe containing additional candidate
        samples.
        total_samples (int): Desired total number of samples in the balanced
        dataset.
        complete_label_column (str): Column name in `complete_df` containing
        sample labels.
        current_label_column (str): Column name in `current_df` containing
        sample labels.

    Returns:
        pd.DataFrame: A new dataframe with balanced classes and shuffled
        samples.

    Raises:
        ValueError: If the number of samples in `complete_df` exceeds
        `total_samples`.
    """
    if len(current_df) > total_samples:
        raise ValueError(
            "The number of samples in the fixed dataset exceeds the desired "
            "total."
        )

    # Remove samples from `complete_df` that are already present in
    # `complete_df` to avoid duplicates
    complete_df = complete_df[~complete_df["ID"].isin(current_df["ID"])]

    # Compute the number of additional samples needed for each class
    remaining_r = total_samples // 2 - \
        len(current_df[current_df[current_label_column] == 'R'])
    remaining_s = total_samples // 2 - \
        len(current_df[current_df[current_label_column] == 'S'])

    # Filter candidate samples for each class
    df_r = complete_df[complete_df[complete_label_column] == 'R']
    df_s = complete_df[complete_df[complete_label_column] == 'S']

    # Limit the number of additional samples to the required amount
    n_r = min(len(df_r), remaining_r)
    n_s = min(len(df_s), remaining_s)

    # Select the first `n_r` and `n_s` samples
    df_r = df_r.head(n_r) if n_r > 0 else pd.DataFrame()
    df_s = df_s.head(n_s) if n_s > 0 else pd.DataFrame()

    # Combine the fixed and newly selected instances
    balanced_df = pd.concat([current_df, df_r, df_s]).reset_index(drop=True)

    return balanced_df


def update_metadata(complete_metadata: str, current_metadata: str,
                    total_of_samples: int, complete_column: str,
                    current_column: str) -> None:
    """
    Updates the metadata table with new samples. Adds the same number of R and
    S samples respecting the desire total of samples.

    Args:
        complete_metadata (str): Complete metadata table path.
        current_metadata (str): Current metadata table path.
        total_of_samples (int): Desired total of samples to keep in the new
        metadata table.
        complete_column (str): Complete metadata column with the samples
        labels.
        current_column (str): Current metadata column with the samples labels.
    """

    complete_df = read_tsv(complete_metadata)
    current_df = read_tsv(current_metadata)

    complete_df.rename(columns={"ID ": "ID", "Unnamed: 2": complete_column},
                       inplace=True)
    complete_df[complete_column] = complete_df[complete_column].fillna(
        complete_df[current_column])
    complete_df.drop_duplicates(inplace=True, ignore_index=True)
    complete_df["ID"] = complete_df["ID"].apply(lambda x: f"{x}.tsv")

    df = balance_samples(complete_df, current_df, total_of_samples,
                         complete_column, current_column)
    df.sort_values(by=[current_column, "ID"], inplace=True)
    print(df[current_column].value_counts())
    df.drop(columns=complete_column, inplace=True)
    df.to_csv(current_metadata, sep="\t", index=False)
