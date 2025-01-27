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
