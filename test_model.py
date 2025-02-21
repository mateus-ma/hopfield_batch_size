import hydra
import torch
from shutil import rmtree
from omegaconf import DictConfig
from deeprc.training import evaluate
from os import path, environ, listdir
from cabgen_hopfield_main import create_task_definition
from src.utils.handle_files import get_most_recent_folder
from widis_lstm_tools.utils.collection import SaverLoader
from src.utils.handle_processing import make_full_dataloader
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, \
    AttentionNetwork, OutputNetwork

environ["PYTORCH_CUDA_ALLOC_CONF"] = ("garbage_collection_threshold:0.8,"
                                      "max_split_size_mb:128,"
                                      "expandable_segments:True")
environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="config", config_name="config")
def test_model(cfg: DictConfig):
    # Test config
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model_path = cfg.test.model_path
    metadata_file = path.abspath(cfg.test.metadata_file)
    orfs_path = path.abspath(cfg.test.orfs_path)

    # CNN config
    kernel_size = cfg.model.kernel_size
    n_kernels = cfg.model.n_kernels
    cnn_layers = cfg.model.sequence_embedding.n_layers

    # Attention config
    attention_layers = cfg.model.attention.n_layers
    attention_units = cfg.model.attention.n_units
    
    # Output config
    output_layers = cfg.model.output.n_layers
    output_units = cfg.model.output.n_units

    multiclass_targets = cfg.task.targets[0].get("possible_target_values")
    binary_targets = cfg.task.targets[0].get("positive_class")
    n_output_features = len(multiclass_targets) \
        if multiclass_targets is not None else len(binary_targets)
    print(f"Using device: {device}")

    print("Reconstructing the model architecture...")
    sequence_embedding_network = SequenceEmbeddingCNN(
        n_input_features=20+3, kernel_size=kernel_size,
        n_kernels=n_kernels, n_layers=cnn_layers)
    attention_network = AttentionNetwork(
        n_input_features=n_kernels, n_layers=attention_layers,
        n_units=attention_units)
    output_network = OutputNetwork(
        n_input_features=n_kernels,
        n_output_features=n_output_features,
        n_layers=output_layers, n_units=output_units)

    model = DeepRC(max_seq_len=13100,
                   sequence_embedding_network=sequence_embedding_network,
                   attention_network=attention_network,
                   output_network=output_network,
                   consider_seq_counts=False, n_input_features=20,
                   add_positional_information=True,
                   sequence_reduction_fraction=0.1, reduction_mb_size=int(5e4),
                   device=device).to(device)

    save_dir = "tmp"
    state = dict(model=model)
    saver_loader = SaverLoader(save_dict=state, device=cfg.device,
                               save_dir=save_dir)

    # Load model state
    if "/" not in model_path or not path.exists(model_path):
        model_folder = path.join(get_most_recent_folder(model_path),
                                 "checkpoint")
        model_path = [path.join(model_folder, file)
                      for file in listdir(model_folder)][0]

    print(f"Loading model from {model_path}...")
    saver_loader.load_from_file(loadname=model_path, verbose=True)
    print("Model loaded successfully.")

    # Set model to evaluation mode
    model.eval()

    print("Preparing the test data...")
    task_definition = create_task_definition(
        cfg.task).to(device=cfg.device)

    testset_eval = make_full_dataloader(
        task_definition=task_definition, metadata_file=metadata_file,
        repertoiresdata_path=orfs_path)

    print("Evaluating the model on test data...")
    scores = evaluate(model=model, dataloader=testset_eval,
                      task_definition=task_definition, device=device)

    print("\nTest set results:")
    print(scores)
    rmtree(save_dir)


if __name__ == "__main__":
    test_model()
