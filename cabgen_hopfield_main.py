import torch
import argparse
import numpy as np
from os import path
from deeprc.training import train, evaluate
from src.utils.create_orf_table import create_orf_table
from deeprc.task_definitions import TaskDefinition, MulticlassTarget
from deeprc.dataset_readers import make_dataloaders, no_sequence_count_scaling
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, \
    AttentionNetwork, OutputNetwork


parser = argparse.ArgumentParser()
parser.add_argument("--n_updates", help=(
    "Number of updates to train for. "
    "Recommended: int(1e5). Default: int(1e3)"),
    type=int, default=int(1e3))
parser.add_argument("--evaluate_at", help=(
    "Evaluate model on training and validation set every `evaluate_at` "
    "updates. This will also check for a new best model for early stopping. "
    "Recommended: int(5e3). Default: int(1e2)."),
    type=int, default=int(1e2))
parser.add_argument("--kernel_size", help=(
    "Size of 1D-CNN kernels (=how many sequence characters a "
    "CNN kernel spans). Default: 9"),
    type=int, default=9)
parser.add_argument("--n_kernels", help=(
    "Number of kernels in the 1D-CNN. This is an important hyper-parameter. "
    "Default: 32"),
    type=int, default=32)
parser.add_argument("--sample_n_sequences", help=(
    "Number of instances to reduce repertoires to during training via"
    "random dropout. This should be less than the number of instances per "
    "repertoire. Only applied during training, not for evaluation. "
    "Default: int(1e4)"),
    type=int, default=int(1e4))
parser.add_argument("--learning_rate", help=(
    "Learning rate of DeepRC using Adam optimizer. Default: 1e-4"),
    type=float, default=1e-4)
parser.add_argument("--device", help=(
    "Device to use for NN computations, as passed to `torch.device()`. "
    "Default: 'cuda:0'."),
    type=str, default="cuda:0")
parser.add_argument("--rnd_seed", help=(
    "Random seed to use for PyTorch and NumPy. Results will still be "
    "non-deterministic due to multiprocessing but weight initialization will "
    "be the same). Default: 0."),
    type=int, default=0)
parser.add_argument("--create_orfs", help=("Create ORFs from JSONs in jsons "
                                           "directory."),
                    type=str, default="False")
args = parser.parse_args()

if args.create_orfs == "True" or args.create_orfs == "true":
    create_orf_table()

# Set computation device
device = torch.device(args.device)
# Set random seed (Weight initialization will be the same)
torch.manual_seed(args.rnd_seed)
np.random.seed(args.rnd_seed)

# Create Task definitions

# Assume we want to train on 1 main task and 5 auxiliary tasks. We will set
# the task-weight of the main task to 1 and of the auxiliary tasks to 0.1/5.
# The tasks-weight is used to compute the training loss as weighted sum of the
# individual tasks losses.
# aux_task_weight = 0.1 / 5
# Below we define how the tasks should be extracted from the metadata file. We
# can choose between combinations of binary, regression, and multiclass tasks.
# The column names have to be found in the metadata file.
task_definition = TaskDefinition(targets=[  # Combines our sub-tasks
    # Add multiclass classification task with softmax output function
    # (=classes mutually exclusive)
    MulticlassTarget(
        column_name="MEM",  # Column name of task in metadata file
        # Values in task column to expect
        possible_target_values=["R", "I", "S"],
        # Weight individual classes (e.g. if class "S" is overrepresented)
        class_weights=[1., 1., 1.],
        # Weight of this task for the total training loss
        task_weight=1
    )
]).to(device=device)

# Get dataset

# Get data loaders for training set and training-, validation-,
# and test-set in evaluation mode (=no random subsampling)
trainingset, trainingset_eval, \
    validationset_eval, testset_eval = make_dataloaders(
        task_definition=task_definition,
        metadata_file=path.abspath("database/metadata.tsv"),
        repertoiresdata_path=path.abspath("database/orfs"),
        metadata_file_id_column="ID",
        sequence_column="orf",
        sequence_counts_column="templates",
        sample_n_sequences=args.sample_n_sequences,
        # Alternative: deeprc.dataset_readers.log_sequence_count_scaling
        sequence_counts_scaling_fn=no_sequence_count_scaling
    )

# Create DeepRC Network

# Create sequence embedding network (for CNN, kernel_size and n_kernels are
# important hyper-parameters)
sequence_embedding_network = SequenceEmbeddingCNN(n_input_features=20+3,
                                                  kernel_size=args.kernel_size,
                                                  n_kernels=args.n_kernels,
                                                  n_layers=1)
# Create attention network
attention_network = AttentionNetwork(
    n_input_features=args.n_kernels, n_layers=2, n_units=32)
# Create output network
output_network = OutputNetwork(
    n_input_features=args.n_kernels,
    n_output_features=task_definition.get_n_output_features(),
    n_layers=1, n_units=32)
# Combine networks to DeepRC network
model = DeepRC(max_seq_len=30,
               sequence_embedding_network=sequence_embedding_network,
               attention_network=attention_network,
               output_network=output_network,
               consider_seq_counts=False, n_input_features=20,
               add_positional_information=True,
               sequence_reduction_fraction=0.1, reduction_mb_size=int(5e4),
               device=device).to(device=device)

# Train DeepRC model
train(model, task_definition=task_definition,
      trainingset_dataloader=trainingset,
      trainingset_eval_dataloader=trainingset_eval,
      learning_rate=args.learning_rate,
      # Get model that performs best for this task
      early_stopping_target_id="MEM",
      validationset_eval_dataloader=validationset_eval,
      n_updates=args.n_updates, evaluate_at=args.evaluate_at,
      # Here our results and trained models will be stored
      device=device, results_directory="results"
      )


scores = evaluate(model=model, dataloader=testset_eval,
                  task_definition=task_definition, device=device)
print(f"Test scores:\n{scores}")
