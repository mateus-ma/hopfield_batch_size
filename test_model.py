import torch
import argparse
from os import environ
from shutil import rmtree
from deeprc.training import evaluate
from widis_lstm_tools.utils.collection import SaverLoader
from src.utils.handle_processing import make_full_dataloader
from deeprc.task_definitions import TaskDefinition, MulticlassTarget
from deeprc.architectures import DeepRC, SequenceEmbeddingCNN, \
    AttentionNetwork, OutputNetwork

environ["PYTORCH_CUDA_ALLOC_CONF"] = ("garbage_collection_threshold:0.8,"
                                      "max_split_size_mb:128")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--device", help="Device for computation: 'cuda:0' or 'cpu'.", type=str,
    default="cuda:0")
parser.add_argument(
    "--model_path", help="Path to the saved model (.pkl).",
    type=str, required=True)
parser.add_argument(
    "--metadata_file", help="Path to the metadata table (TSV).", type=str,
    required=True)
parser.add_argument(
    "--orfs_path", help="Path to the folder with ORFs.", type=str,
    required=True)
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Reconstructing the model architecture...")
sequence_embedding_network = SequenceEmbeddingCNN(
    n_input_features=20+3, kernel_size=9, n_kernels=32, n_layers=1)
attention_network = AttentionNetwork(
    n_input_features=32, n_layers=2, n_units=32)
output_network = OutputNetwork(
    n_input_features=32, n_output_features=3, n_layers=1, n_units=32)

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
saver_loader = SaverLoader(save_dict=state, device=args.device,
                           save_dir=save_dir)

# Load model state
print(f"Loading model from {args.model_path}...")
saver_loader.load_from_file(loadname=args.model_path, verbose=True)
print("Model loaded successfully.")

# Set model to evaluation mode
model.eval()

print("Preparing the test data...")
task_definition = TaskDefinition(targets=[
    MulticlassTarget(
        column_name="MEM",
        possible_target_values=["R", "I", "S"],
        class_weights=[1., 1., 1.],
        task_weight=1
    )
]).to(device=device)

testset_eval = make_full_dataloader(
    task_definition=task_definition, metadata_file=args.metadata_file,
    repertoiresdata_path=args.orfs_path)

print("Evaluating the model on test data...")
scores = evaluate(model=model, dataloader=testset_eval,
                  task_definition=task_definition, device=device)

print("\nTest set results:")
print(scores)
rmtree(save_dir)
