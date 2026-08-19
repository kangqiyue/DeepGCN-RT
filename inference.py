"""Command-line inference for the released DeepGCN-RT checkpoint."""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

import torch

from dataset import feature_to_dgl_graph, get_edge_dim, get_node_dim, smiles2graph
from models import GCNModelWithEdgeAFPreadout


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = REPOSITORY_ROOT / "model_path" / "best_model_weight.pth"


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a requested inference device and fail clearly when unavailable."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(device)


def build_model() -> GCNModelWithEdgeAFPreadout:
    """Build the architecture used by the released 16-layer checkpoint."""
    return GCNModelWithEdgeAFPreadout(
        node_in_dim=get_node_dim(),
        edge_in_dim=get_edge_dim(),
        hidden_feats=[200] * 16,
        dropout=0.1,
    )


def load_pretrained_model(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    device: Optional[torch.device] = None,
) -> GCNModelWithEdgeAFPreadout:
    """Load the released checkpoint onto ``device`` in evaluation mode."""
    selected_device = device or resolve_device()
    checkpoint_path = Path(model_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model = build_model()
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=selected_device,
            weights_only=True,
        )
    except TypeError:  # PyTorch < 2.0 does not support weights_only.
        checkpoint = torch.load(checkpoint_path, map_location=selected_device)

    model.load_state_dict(checkpoint)
    model.to(selected_device)
    model.eval()
    return model


def predict_retention_time(
    smiles: str,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    device: str = "auto",
) -> float:
    """Predict retention time in seconds for one SMILES string."""
    selected_device = resolve_device(device)
    graph = feature_to_dgl_graph(smiles2graph(smiles)).to(selected_device)
    model = load_pretrained_model(model_path=model_path, device=selected_device)

    with torch.no_grad():
        output = model(graph)
    return float(output.reshape(-1)[0].detach().cpu().item())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict chromatographic retention time with DeepGCN-RT."
    )
    parser.add_argument(
        "--smiles",
        "--SMILES",
        dest="smiles",
        required=True,
        help="SMILES string for one molecule.",
    )
    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Checkpoint path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        prediction = predict_retention_time(
            smiles=args.smiles,
            model_path=args.model_path,
            device=args.device,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"SMILES: {args.smiles}")
    print(f"Predicted retention time: {prediction:.4f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
