import pandas as pd
from typing import List, Union
from collections import Counter
import src.utils.handle_files as hf


def process_json(json_file: str) -> List[List[Union[str, int]]]:
    data = hf.read_json(json_file)
    features = data.get("features", [])

    sequences = []
    rows = []
    for feature in features:
        locus = feature.get("locus", "")
        seq_type = feature.get("type", "")
        product = feature.get("product", "")
        sequence = feature.get("nt", "")

        sequences.append(sequence)

        row = [locus, seq_type, product, sequence]
        rows.append(row)

    seq_counter = Counter(sequences)
    for row in rows:
        seq_counting = seq_counter.get(row[3])
        row.append(seq_counting)

    return rows


def process_tsv(df: pd.DataFrame) -> pd.DataFrame:
    columns = {"sequence": "orf", "count": "templates"}

    selected_df = df[["sequence", "count"]].copy()
    selected_df.rename(columns=columns, inplace=True)  # type: ignore
    return selected_df  # type: ignore
