import hydra
import torch
import pandas as pd
from tqdm import tqdm
from shutil import rmtree
from os import path, listdir
from omegaconf import DictConfig
from cabgen_hopfield_main import create_task_definition
from src.utils.handle_files import get_most_recent_folder
from widis_lstm_tools.utils.collection import SaverLoader
from src.utils.handle_processing import make_full_dataloader, \
    insert_datetime_into_filename
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, \
    AttentionNetwork, OutputNetwork


@hydra.main(version_base=None, config_path="config", config_name="config")
def predict_samples(cfg: DictConfig):
    """Generate predictions for test samples and save them as a table."""

    # Device configuration
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Input paths
    model_path = cfg.test.model_path
    metadata_file = path.abspath(cfg.test.metadata_file)
    orfs_path = path.abspath(cfg.test.orfs_path)

    # Model architecture configuration
    kernel_size = cfg.model.kernel_size
    n_kernels = cfg.model.n_kernels
    cnn_layers = cfg.model.sequence_embedding.n_layers
    attention_layers = cfg.model.attention.n_layers
    output_layers = cfg.model.output.n_layers
    multiclass_targets = cfg.task.targets[0].get("possible_target_values")
    binary_targets = cfg.task.targets[0].get("positive_class")
    n_output_features = len(multiclass_targets) \
        if multiclass_targets is not None else len(binary_targets)

    # Get the antibiotic name to use as the column name
    antibiotic_name = cfg.task.targets[0].column_name

    print(f"Using device: {device}")

    # Reconstructing the model architecture
    print("Reconstructing the model architecture...")
    sequence_embedding_network = SequenceEmbeddingCNN(
        n_input_features=20+3, kernel_size=kernel_size,
        n_kernels=n_kernels, n_layers=cnn_layers)
    attention_network = AttentionNetwork(
        n_input_features=n_kernels, n_layers=attention_layers,
        n_units=n_kernels)
    output_network = OutputNetwork(
        n_input_features=n_kernels,
        n_output_features=n_output_features,
        n_layers=output_layers, n_units=n_kernels)

    model = DeepRC(
        max_seq_len=13100,
        sequence_embedding_network=sequence_embedding_network,
        attention_network=attention_network,
        output_network=output_network,
        consider_seq_counts=False, n_input_features=20,
        add_positional_information=True,
        sequence_reduction_fraction=0.1, reduction_mb_size=int(5e4),
        device=device
    ).to(device)

    # Load saved model weights
    print("Loading model...")
    output_dir = "tmp"
    state = dict(model=model)
    saver_loader = SaverLoader(save_dict=state, device=cfg.device,
                               save_dir=output_dir)

    if "/" not in model_path or not path.exists(model_path):
        model_folder = path.join(
            get_most_recent_folder(model_path), "checkpoint")
        model_path = [path.join(model_folder, file)
                      for file in listdir(model_folder)][0]

    saver_loader.load_from_file(loadname=model_path, verbose=True)
    print(f"Model loaded successfully from {model_path}.")

    # Set the model to evaluation mode
    model.eval()

    # Prepare test data
    print("Preparing test data...")
    task_definition = create_task_definition(cfg.task).to(device=device)

    testset_eval = make_full_dataloader(
        task_definition=task_definition,
        metadata_file=metadata_file,
        repertoiresdata_path=orfs_path
    )

    print("Generating predictions...")
    all_predictions = []

    # Iterate over test data
    with torch.no_grad():
        for scoring_data in tqdm(testset_eval, total=len(testset_eval),
                                 desc="Evaluating model"):

            # Extract batch data
            targets, inputs, sequence_lengths, \
                counts_per_sequence, sample_ids = scoring_data

            # Apply attention-based sequence reduction and create a minibatch
            _, inputs, sequence_lengths, n_sequences = \
                model.reduce_and_stack_minibatch(
                    targets, inputs, sequence_lengths, counts_per_sequence)

            # Forward pass to obtain predictions
            raw_outputs = model(inputs_flat=inputs,
                                sequence_lengths_flat=sequence_lengths,
                                n_sequences_per_bag=n_sequences)
            probabilities = torch.sigmoid(raw_outputs).cpu().numpy().flatten()

            # Store the results
            for sample_id, prob in zip(sample_ids, probabilities):
                all_predictions.append(
                    {"ID": sample_id, antibiotic_name: prob})

    # Create DataFrame and save as TSV
    df = pd.DataFrame(all_predictions)
    output_file = insert_datetime_into_filename(
        f"{path.basename(model_path).split('.')[0]}_predict.tsv")
    df.to_csv((output_file), index=False, sep="\t")

    print(f"Predictions saved to {output_file}")
    rmtree(output_dir)


if __name__ == "__main__":
    predict_samples()
