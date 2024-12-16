import torch
import argparse
import numpy as np
from os import path
from deeprc.training import evaluate
from deeprc.task_definitions import TaskDefinition, MulticlassTarget
from deeprc.dataset_readers import make_dataloaders, no_sequence_count_scaling

parser = argparse.ArgumentParser()
parser.add_argument("--device", help=(
    "Device to use for NN computations, as passed to `torch.device()`. "
    "Default: 'cuda:0'."),
    type=str, default="cuda:0")
parser.add_argument("--rnd_seed", help=(
    "Random seed to use for PyTorch and NumPy. Results will still be "
    "non-deterministic due to multiprocessing but weight initialization will "
    "be the same). Default: 0."),
    type=int, default=0)
parser.add_argument("--model_path", help=(
    "Path to the saved model file in pickle format."),
    type=str, required=True)
args = parser.parse_args()

device = torch.device(args.device)
torch.manual_seed(args.rnd_seed)
np.random.seed(args.rnd_seed)

task_definition = TaskDefinition(targets=[
    MulticlassTarget(
        column_name="MEM",
        possible_target_values=["R", "I", "S"],
        class_weights=[1., 1., 1.],
        task_weight=1
    )
]).to(device=device)

_, _, _, testset_eval = make_dataloaders(
    task_definition=task_definition,
    metadata_file=path.abspath("database/metadata.tsv"),
    repertoiresdata_path=path.abspath("database/orfs"),
    metadata_file_id_column="ID",
    sequence_column="orf",
    sequence_counts_column="templates",
    # Carregar todos os dados
    sample_n_sequences=None,  # type: ignore
    sequence_counts_scaling_fn=no_sequence_count_scaling
)

print(f"Loading model from {args.model_path}")
with open(args.model_path, "rb") as f:
    model = torch.load(f)

model.eval()

scores = evaluate(model=model, dataloader=testset_eval,
                  task_definition=task_definition, device=device)
print(f"Test scores:\n{scores}")
