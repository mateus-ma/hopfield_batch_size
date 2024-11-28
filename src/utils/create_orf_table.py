import pandas as pd
from os import path, listdir, mkdir
from src.utils.handle_files import write_table, read_tsv
from src.utils.handle_processing import process_json, process_tsv


def create_orf_table():
    metadata_file = path.abspath("database/metadata.tsv")
    metadata = pd.read_csv(metadata_file, sep='\t')
    ids_to_use = {id.split(".")[0] for id in metadata['ID']}

    complete_table_dir = path.abspath("database/complete_tables")
    if not path.exists(complete_table_dir):
        mkdir(complete_table_dir)

    base_dir = path.abspath("jsons")
    json_files = [path.join(base_dir, json) for json in listdir(base_dir)
                  if json.endswith("json") and
                  json.split(".json")[0] in ids_to_use]

    if len(json_files) == 0:
        return

    for json_file in json_files:
        rows = process_json(json_file)

        table_name = f"{json_file.split('/')[-1].split('.')[0]}.tsv"
        complete_table = path.join(complete_table_dir, table_name)

        write_table(complete_table, rows)

        orf_table = path.abspath(path.join("database/orfs", table_name))
        df = read_tsv(complete_table)

        orf_df = process_tsv(df)
        write_table(orf_table, orf_df)
