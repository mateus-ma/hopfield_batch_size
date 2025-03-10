import torch
from omegaconf import DictConfig
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, \
    AttentionNetwork, OutputNetwork


def build_model(cfg: DictConfig) -> DeepRC:
    """
    Constructs and returns a DeepRC model based on the given configuration.

    Args:
        cfg (DictConfig): Configuration object containing model parameters,
        including CNN, attention, and output network settings.

    Returns:
        DeepRC: A DeepRC model instance with the specified architecture and
        parameters.
    """

    # Select the device (GPU if available, otherwise CPU)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # CNN (Convolutional Neural Network) configuration
    kernel_size = cfg.model.kernel_size  # Size of convolutional kernels
    n_kernels = cfg.model.n_kernels  # Number of convolutional filters
    cnn_layers = cfg.model.sequence_embedding.n_layers  # Number of CNN layers

    # Attention mechanism configuration

    # Number of attention layers
    attention_layers = cfg.model.attention.n_layers
    # Number of attention units per layer
    attention_units = cfg.model.attention.n_units
    # Number of attention heads
    attention_heads = cfg.model.attention.n_heads

    # Output network configuration
    # Number of fully connected layers
    output_layers = cfg.model.output.n_layers
    # Number of units in each output layer
    output_units = cfg.model.output.n_units

    # Determine the number of output features based on the type of
    # classification task
    multiclass_targets = cfg.task.targets[0].get(
        "possible_target_values")  # Multi-class labels
    binary_targets = cfg.task.targets[0].get(
        "positive_class")  # Binary classification target

    # Number of output features depends on the classification type
    n_output_features = len(multiclass_targets) if multiclass_targets is not \
        None else len(binary_targets)

    print(f"Using device: {device}")
    print("Reconstructing the model architecture...")

    # Define the sequence embedding network using a 1D CNN
    sequence_embedding_network = SequenceEmbeddingCNN(
        # 20 standard amino acid features + 3 additional features
        n_input_features=20+3,
        kernel_size=kernel_size,
        n_kernels=n_kernels,
        n_layers=cnn_layers
    )

    # Define the attention network for sequence feature aggregation
    attention_network = AttentionNetwork(
        n_input_features=n_kernels,  # Output from CNN
        n_layers=attention_layers,
        n_units=attention_units,
        n_heads=attention_heads
    )

    # Define the output network responsible for final classification
    output_network = OutputNetwork(
        n_input_features=n_kernels,  # Output from attention mechanism
        n_output_features=n_output_features,  # Number of classes
        n_layers=output_layers,
        n_units=output_units
    )

    # Construct the DeepRC model with the defined components
    model = DeepRC(
        max_seq_len=5000,  # Maximum sequence length supported by the model
        sequence_embedding_network=sequence_embedding_network,
        attention_network=attention_network,
        output_network=output_network,
        consider_seq_counts=False,  # Ignore sequence count information
        n_input_features=20,  # Number of input features per residue
        add_positional_information=True,  # Include positional encodings
        # Reduce sequence length by 10% during processing
        sequence_reduction_fraction=0.1,
        # Batch size for memory-efficient processing
        reduction_mb_size=int(5e4),
        device=device  # Device to run the model on (CPU/GPU)
    ).to(device)

    return model


def clear_gpu_memory():
    """Release unused VRAM before and after training."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
