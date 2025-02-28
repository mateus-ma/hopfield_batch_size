import pandas as pd
from datetime import datetime
from collections import Counter
import src.utils.handle_files as hf
from torch.utils.data import DataLoader
from typing import List, Union, Callable
from deeprc.task_definitions import TaskDefinition
from deeprc.dataset_converters import DatasetToHDF5
from deeprc.dataset_readers import RepertoireDataset, no_stack_collate_fn, \
    no_sequence_count_scaling


def process_json(json_file: str) -> List[List[Union[str, int]]]:
    """
    Reads a JSON file and extracts specific genomic features.

    This function processes a JSON file containing genomic data and extracts
    relevant information such as locus, type, product, and nucleotide sequence.
    Additionally, it counts the occurrences of each unique sequence.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        List[List[Union[str, int]]]: A list of lists where each sublist
        contains:
            - locus (str)
            - feature type (str)
            - product (str)
            - nucleotide sequence (str)
            - sequence count (int)
    """
    # Read JSON data
    data = hf.read_json(json_file)
    features = data.get("features", [])

    sequences = []
    rows = []

    # Extract relevant information from each feature
    for feature in features:
        locus = feature.get("locus", "")
        seq_type = feature.get("type", "")
        product = feature.get("product", "")
        sequence = feature.get("nt", "")

        sequences.append(sequence)

        row = [locus, seq_type, product, sequence]
        rows.append(row)

    # Count occurrences of each unique sequence
    seq_counter = Counter(sequences)

    # Append sequence count to each row
    for row in rows:
        seq_counting = seq_counter.get(row[3])
        row.append(seq_counting)

    return rows


def process_tsv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a TSV-formatted DataFrame by selecting and renaming specific
    columns.

    This function extracts the "sequence" and "count" columns from the given
    DataFrame,
    renaming them to "orf" and "templates", respectively.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least "sequence" and
        "count" columns.

    Returns:
        pd.DataFrame: Processed DataFrame with renamed columns.
    """
    # Mapping original column names to new names
    columns = {"sequence": "orf", "count": "templates"}

    # Select and rename specified columns
    selected_df = df[["sequence", "count"]].copy()
    selected_df.rename(columns=columns, inplace=True)  # type: ignore

    return selected_df  # type: ignore


def make_full_dataloader(
    task_definition: TaskDefinition, metadata_file: str,
        repertoiresdata_path: str, inputformat: str = 'NCL',
        keep_dataset_in_ram: bool = True,
        sequence_counts_scaling_fn: Callable = no_sequence_count_scaling,
        sequence_column: str = "orf",
        sequence_count_column: str = "templates",
        metadata_file_id_column: str = 'ID',
        metadata_file_column_sep: str = '\t',
        verbose: bool = True) -> DataLoader:
    """
    Creates a DataLoader for the full dataset without splitting or subsampling.

    Parameters
    ----------
    task_definition: TaskDefinition
        TaskDefinition object containing the tasks to train the DeepRC model
        on.
    metadata_file : str
        Filepath of metadata .tsv file with targets.
    repertoiresdata_path : str
        Filepath of hdf5 file containing repertoire sequence data or filepath
        of folder containing the repertoire
        `.tsv`/`.csv` files. `.tsv`/`.csv` will be converted to a hdf5 file.
    batch_size : int
        Number of repertoires per minibatch during evaluation or inference.
    inputformat : 'NCL' or 'NLC'
        Format of input feature array;
        'NCL' -> (batchsize, channels, seq.length);
        'LNC' -> (seq.length, batchsize, channels);
    keep_dataset_in_ram : bool
        It is faster to load the full hdf5 file into the RAM instead of
        keeping it on the disk.
        If False, the hdf5 file will be read from the disk and consume less
        RAM.
    sequence_counts_scaling_fn
        Scaling function for sequence counts. E.g.
        `deeprc.dataset_readers.log_sequence_count_scaling` or
        `deeprc.dataset_readers.no_sequence_count_scaling`.
    metadata_file_id_column : str
        Name of column holding the repertoire names in `metadata_file`.
    metadata_file_column_sep : str
        The column separator in `metadata_file`.
    verbose : bool
        Activate verbose mode.

    Returns
    ---------
    full_dataloader: DataLoader
        Dataloader for the entire dataset without any splitting or subsampling.
    """
    try:
        hdf5_file = repertoiresdata_path + ".hdf5"
        if verbose:
            print(f"Converting: {repertoiresdata_path}\n->\n{hdf5_file}")
        converter = DatasetToHDF5(
            repertoiresdata_directory=repertoiresdata_path,
            sequence_column=sequence_column,
            sequence_counts_column=sequence_count_column,
            column_sep='\t',
            filename_extension='.tsv',
            h5py_dict=None, verbose=verbose)  # type:ignore
        converter.save_data_to_file(output_file=hdf5_file, n_workers=4)
        if verbose:
            print(f"\tSuccessfully created {hdf5_file}!")

        if verbose:
            print(f"Loading full dataset from {hdf5_file}")
        full_dataset = RepertoireDataset(
            metadata_filepath=metadata_file,
            hdf5_filepath=hdf5_file,
            sample_id_column=metadata_file_id_column,
            metadata_file_column_sep=metadata_file_column_sep,
            task_definition=task_definition, keep_in_ram=keep_dataset_in_ram,
            inputformat=inputformat,
            sequence_counts_scaling_fn=sequence_counts_scaling_fn)

        full_dataloader = DataLoader(
            full_dataset, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=no_stack_collate_fn)
        if verbose:
            print(f"Full dataset contains {len(full_dataloader)} samples.")
        return full_dataloader
    except Exception as error:
        raise ValueError(f"Can't return data...\n{error}")


def get_datetime() -> str:
    """
    Generates a timestamp string with the current date and time.

    The format used is: YYYY_MM_DD_HH_MM_SS.

    Returns:
        str: The formatted timestamp.
    """
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def insert_datetime_into_filename(filename: str) -> str:
    """
    Inserts a timestamp into a filename before the file extension.

    Example:
        Input: "results.csv"
        Output: "results_2025_02_28_14_30_00.csv"

    Args:
        filename (str): The original filename.

    Returns:
        str: The filename with the timestamp inserted.
    """
    # Split the filename to extract the name and extension
    name, extension = filename.rsplit(".", 1)

    # Append the timestamp before the extension
    return f"{name}_{get_datetime()}.{extension}"


def calculate_accuracy(df: pd.DataFrame) -> float:
    """
    Calculates the accuracy of predictions based on the given dataframe.

    Accuracy is computed as the proportion of correctly predicted labels.

    Args:
        df (pd.DataFrame): Dataframe containing the columns 'Prediction' and
        'Label'.

    Returns:
        float: The accuracy score (between 0 and 1).
    """
    if "Prediction" not in df.columns or "Label" not in df.columns:
        raise ValueError("The dataframe must contain 'Prediction' and 'Label' "
                         "columns.")

    correct_predictions = (df["Prediction"] == df["Label"]).sum()
    total_samples = len(df)

    return float(correct_predictions / total_samples) if total_samples > 0 \
        else 0.0
