import hydra
import torch
import numpy as np
from os import path, environ
from omegaconf import DictConfig
from deeprc.training import train, evaluate
from cabgen_hopfield_main import create_task_definition
from src.utils.handle_machine_learning import build_model, clear_gpu_memory
from deeprc.dataset_readers import make_dataloaders_stratified, \
    no_sequence_count_scaling


environ["PYTORCH_CUDA_ALLOC_CONF"] = ("garbage_collection_threshold:0.8,"
                                      "max_split_size_mb:128,"
                                      "expandable_segments:True")
environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Trains and evaluates a DeepRC model using cross-validation.

    This function:
    - Loads configuration settings from Hydra.
    - Prepares the dataset and model for training.
    - Performs 5-fold cross-validation.
    - Evaluates the model on each fold and selects the best-performing one.

    Args:
        cfg (DictConfig): Configuration object containing dataset paths,
            model hyperparameters, training settings, and other relevant
            parameters.
    """

    # Set device for computation
    device = torch.device(cfg.device)

    # Load training parameters
    n_updates = cfg.training.n_updates
    evaluate_at = cfg.training.evaluate_at
    sample_n_sequences = cfg.data_splitting.sample_n_sequences
    learning_rate = cfg.training.learning_rate
    batch_size = cfg.training.batch_size
    rnd_seed = cfg.rnd_seed
    results_directory = cfg.results_directory

    # Load dataset file paths
    metadata_file = path.abspath(cfg.database.metadata_file)
    repertoiresdata_path = path.abspath(cfg.database.repertoiresdata_path)

    # Load dataset split settings
    stratify = cfg.data_splitting.stratify
    column_name = cfg.task.targets[0]["column_name"]
    metadata_file_id_column = cfg.data_splitting.metadata_file_id_column
    sequence_column = cfg.data_splitting.sequence_column
    sequence_counts_column = cfg.data_splitting.sequence_counts_column

    # Set random seeds for reproducibility
    torch.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)

    # Define the prediction task
    task_definition = create_task_definition(cfg.task).to(device=cfg.device)

    best_fold = None
    best_score = float('-inf')

    # Perform 5-fold cross-validation
    for fold_idx in range(5):
        print(f"Training on fold {fold_idx+1}/5...")

        # Create stratified data loaders for the current fold
        trainingset, trainingset_eval, validationset_eval, _ = \
            make_dataloaders_stratified(
                task_definition=task_definition,
                metadata_file=metadata_file,
                repertoiresdata_path=repertoiresdata_path,
                n_splits=5,
                cross_validation_fold=fold_idx,
                stratify=stratify,
                rnd_seed=rnd_seed,
                batch_size=batch_size,
                metadata_file_id_column=metadata_file_id_column,
                sequence_column=sequence_column,
                sequence_counts_column=sequence_counts_column,
                sample_n_sequences=sample_n_sequences,
                sequence_counts_scaling_fn=no_sequence_count_scaling,
                active_cv=True
            )

        # Build the DeepRC model
        model = build_model(cfg)

        # Train the model on the current fold
        train(
            model,
            task_definition=task_definition,
            trainingset_dataloader=trainingset,
            trainingset_eval_dataloader=trainingset_eval,
            learning_rate=learning_rate,
            early_stopping_target_id=column_name,
            validationset_eval_dataloader=validationset_eval,
            n_updates=n_updates,
            evaluate_at=evaluate_at,
            device=device,
            results_directory=results_directory  # type: ignore
        )

        # Evaluate the model on the validation set
        scores = evaluate(
            model=model,
            dataloader=validationset_eval,
            task_definition=task_definition,
            device=device
        )
        print(f"Validation scores for fold {fold_idx+1}: {scores}")
        clear_gpu_memory()

        # Compute a custom score to determine the best fold
        current_score = (
            scores.get("roc_auc", 0) +
            scores.get("bacc", 0) +
            scores.get("f1", 0) -
            scores.get("loss", 0)
        )

        if current_score > best_score:
            best_score = current_score
            best_fold = fold_idx+1

    print(f"The best fold is {best_fold}.")


if __name__ == "__main__":
    main()
