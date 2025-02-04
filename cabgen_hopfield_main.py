import hydra
import torch
import numpy as np
from os import path, environ
from omegaconf import DictConfig
from deeprc.training import train, evaluate
from src.utils.create_orf_table import create_orf_table
from deeprc.task_definitions import TaskDefinition, MulticlassTarget, \
    BinaryTarget, RegressionTarget
from deeprc.dataset_readers import make_dataloaders_stratified, \
    no_sequence_count_scaling
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, \
    AttentionNetwork, OutputNetwork

environ["PYTORCH_CUDA_ALLOC_CONF"] = ("garbage_collection_threshold:0.8,"
                                      "max_split_size_mb:128,"
                                      "expandable_segments:True")
environ["HYDRA_FULL_ERROR"] = "1"


def create_task_definition(task_config):
    targets = []
    for target in task_config["targets"]:
        task_type = target["type"]
        # Add multiclass classification task with softmax output function
        # (=classes mutually exclusive)
        if task_type == "multiclass":
            targets.append(MulticlassTarget(
                # Column name of task in metadata file
                column_name=target["column_name"],
                # Values in task column to expect
                possible_target_values=target["possible_target_values"],
                # Weight individual classes (e.g. if class "S" is
                # overrepresented)
                class_weights=target["class_weights"],
                # Weight of this task for the total training loss
                task_weight=target["task_weight"]
            ))
        elif task_type == "binary":
            # Add binary classification task with sigmoid output function
            targets.append(BinaryTarget(
                # Column name of task in metadata file
                column_name=target["column_name"],
                # Entries with value '+' will be positive class, others will
                # be negative class
                true_class_value=target["positive_class"],
                # We can up- or down-weight the positive class if the classes
                # are imbalanced
                pos_weight=target["pos_weight"],
                # Weight of this task for the total training loss
                task_weight=target["task_weight"]
            ))
        elif task_type == "regression":
            # Add regression task with linear output function
            targets.append(RegressionTarget(
                # Column name of task in metadata file
                column_name=target["column_name"],
                # Normalize targets by ((target_value - mean) / std)
                normalization_mean=target["normalization_mean"],
                normalization_std=target["normalization_std"],
                # Weight of this task for the total training loss
                task_weight=target["task_weight"]
            ))
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    return TaskDefinition(targets=targets)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Number of updates to train for.
    # Recommended: int(1e5). Default: int(1e3).
    n_updates = cfg.training.n_updates

    # Evaluate model on training and validation set every `evaluate_at`
    # updates. This will also check for a new best model for early stopping.
    # Recommended: int(5e3). Default: int(1e2).
    evaluate_at = cfg.training.evaluate_at

    # Size of 1D-CNN kernels (=how many sequence characters a CNN kernel
    # (spans). Default: 9.
    kernel_size = cfg.model.kernel_size

    # Number of kernels in the 1D-CNN. This is an important hyper-parameter.
    # Default: 32.
    n_kernels = cfg.model.n_kernels

    # Number of instances to reduce repertoires to during training via
    # random dropout. This should be less than the number of instances per
    # repertoire. Only applied during training, not for evaluation.
    # Default: int(1e4).
    sample_n_sequences = cfg.data_splitting.sample_n_sequences

    # Learning rate of DeepRC using Adam optimizer. Default: 1e-4.
    learning_rate = cfg.training.learning_rate

    # Device to use for NN computations, as passed to `torch.device()`.
    # Default: 'cuda:0'.
    device = cfg.device

    # Random seed to use for PyTorch and NumPy.
    # Results will still be non-deterministic due to multiprocessing, but
    # weight initialization will be the same.
    # Default: 0.
    rnd_seed = cfg.rnd_seed

    # Create ORFs from JSONs in the `jsons` directory. Default: False.
    create_orfs = cfg.create_orfs

    # General
    results_directory = cfg.results_directory
    metadata_file = path.abspath(cfg.database.metadata_file)
    repertoiresdata_path = path.abspath(cfg.database.repertoiresdata_path)

    # Data splitting
    stratify = cfg.data_splitting.stratify
    metadata_file_id_column = cfg.data_splitting.metadata_file_id_column
    sequence_column = cfg.data_splitting.sequence_column
    sequence_counts_column = cfg.data_splitting.sequence_counts_column

    # Model
    cnn_layers = cfg.model.sequence_embedding.n_layers
    attention_layers = cfg.model.attention.n_layers
    output_layers = cfg.model.output.n_layers

    if create_orfs:
        create_orf_table()

    # Set computation device
    device = torch.device(device)
    # Set random seed (Weight initialization will be the same)
    torch.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)

    # Create Task definitions

    # Assume we want to train on 1 main task and 5 auxiliary tasks. We will set
    # the task-weight of the main task to 1 and of the auxiliary tasks to
    # 0.1/5. The tasks-weight is used to compute the training loss as weighted
    # sum of the individual tasks losses.
    # aux_task_weight = 0.1 / 5
    # Below we define how the tasks should be extracted from the metadata file.
    # We can choose between combinations of binary, regression, and multiclass
    # tasks. The column names have to be found in the metadata file.
    task_definition = create_task_definition(
        cfg.task).to(device=cfg.device)

    # Get dataset

    # Get data loaders for training set and training-, validation-,
    # and test-set in evaluation mode (=no random subsampling)
    trainingset, trainingset_eval, \
        validationset_eval, testset_eval = make_dataloaders_stratified(
            task_definition=task_definition,
            metadata_file=metadata_file,
            repertoiresdata_path=repertoiresdata_path,
            n_splits=5,
            stratify=stratify,
            rnd_seed=rnd_seed,
            metadata_file_id_column=metadata_file_id_column,
            sequence_column=sequence_column,
            sequence_counts_column=sequence_counts_column,
            sample_n_sequences=sample_n_sequences,
            # Alternative: deeprc.dataset_readers.log_sequence_count_scaling
            sequence_counts_scaling_fn=no_sequence_count_scaling
        )

    # Create DeepRC Network

    # Create sequence embedding network (for CNN, kernel_size and n_kernels are
    # important hyper-parameters)
    sequence_embedding_network = SequenceEmbeddingCNN(n_input_features=20+3,
                                                      kernel_size=kernel_size,
                                                      n_kernels=n_kernels,
                                                      n_layers=cnn_layers)
    # Create attention network
    attention_network = AttentionNetwork(
        n_input_features=n_kernels, n_layers=attention_layers,
        n_units=n_kernels)
    # Create output network
    output_network = OutputNetwork(
        n_input_features=n_kernels,
        n_output_features=task_definition.get_n_output_features(),
        n_layers=output_layers, n_units=n_kernels)
    # Combine networks to DeepRC network
    model = DeepRC(max_seq_len=13100,
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
          learning_rate=learning_rate,
          # Get model that performs best for this task
          early_stopping_target_id="MEM",
          validationset_eval_dataloader=validationset_eval,
          n_updates=n_updates, evaluate_at=evaluate_at,
          # Here our results and trained models will be stored
          device=device, results_directory=results_directory
          )

    scores = evaluate(model=model, dataloader=testset_eval,
                      task_definition=task_definition, device=device)
    print(f"Test scores:\n{scores}")


if __name__ == "__main__":
    main()
