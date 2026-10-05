"""Limited-data GFP/AAV protein fitness benchmark definitions and evaluation.

The GFP/AAV benchmark tasks are data-regime definitions.  Training-time models
should consume only the task-specific subset.  The Kirjner et al. CNN oracle is
kept in this module as an evaluation-only dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from adapters.protein.validation import CANONICAL_AA, validate_sequence


TASK_CONFIG_DIR = Path("configs/protein_fitness_benchmarks")
GGS_ALPHABET = "ARNDCQEGHILKMFPSTWYV"


@dataclass(frozen=True, slots=True)
class ProteinFitnessTask:
    name: str
    protein: str
    training_subset_size: int
    normalized_fitness_median: float
    training_fitness_range: tuple[float, float]
    percentile_definition: str
    fitness_percentile_range: tuple[float, float] | None
    mutation_gap_to_99th_percentile: int
    full_dms_size: int
    oracle_name: str = "kirjner_2023_1d_cnn"
    oracle_evaluation_only: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ProteinFitnessTask":
        training = values.get("training_subset", {})
        if not isinstance(training, Mapping):
            raise ValueError("task config training_subset must be a mapping")
        full_dms = values.get("full_dms", {})
        if not isinstance(full_dms, Mapping):
            raise ValueError("task config full_dms must be a mapping")
        oracle = values.get("oracle", {})
        if not isinstance(oracle, Mapping):
            raise ValueError("task config oracle must be a mapping")
        percentile_range = training.get("fitness_percentile_range")
        percentile_tuple = None
        if percentile_range is not None:
            percentile_tuple = tuple(float(v) for v in percentile_range)
            if len(percentile_tuple) != 2:
                raise ValueError("fitness_percentile_range must contain two values")
        fitness_range = tuple(float(v) for v in training["normalized_fitness_range"])
        if len(fitness_range) != 2:
            raise ValueError("normalized_fitness_range must contain two values")
        return cls(
            name=str(values["name"]),
            protein=str(values["protein"]),
            training_subset_size=int(training["size"]),
            normalized_fitness_median=float(training["normalized_fitness_median"]),
            training_fitness_range=(float(fitness_range[0]), float(fitness_range[1])),
            percentile_definition=str(training["definition"]),
            fitness_percentile_range=percentile_tuple,
            mutation_gap_to_99th_percentile=int(training["min_mutations_to_99th_percentile"]),
            full_dms_size=int(full_dms["size"]),
            oracle_name=str(oracle.get("name", "kirjner_2023_1d_cnn")),
            oracle_evaluation_only=bool(oracle.get("evaluation_only", True)),
        )

    def to_training_config(self) -> dict[str, Any]:
        """Return the subset-only view safe for training-time pipelines."""
        return {
            "name": self.name,
            "protein": self.protein,
            "training_subset": {
                "size": self.training_subset_size,
                "normalized_fitness_median": self.normalized_fitness_median,
                "normalized_fitness_range": list(self.training_fitness_range),
                "definition": self.percentile_definition,
                "fitness_percentile_range": (
                    None
                    if self.fitness_percentile_range is None
                    else list(self.fitness_percentile_range)
                ),
                "min_mutations_to_99th_percentile": self.mutation_gap_to_99th_percentile,
                "training_only": True,
            },
        }


@dataclass(frozen=True, slots=True)
class OracleNormalizationStats:
    protein: str
    full_dms_size: int
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if float(self.y_max) <= float(self.y_min):
            raise ValueError("full-DMS y_max must be greater than y_min")


@dataclass(frozen=True, slots=True)
class ProteinFitnessEvaluation:
    median_normalized_fitness: float
    diversity: float
    novelty: float | None
    raw_fitness: list[float]
    normalized_fitness: list[float]
    num_sequences: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_task_config(path_or_name: str | Path) -> ProteinFitnessTask:
    path = Path(path_or_name)
    if not path.exists():
        path = TASK_CONFIG_DIR / f"{path_or_name}.json"
    values = json.loads(path.read_text())
    return ProteinFitnessTask.from_mapping(values)


def load_all_task_configs(config_dir: str | Path = TASK_CONFIG_DIR) -> dict[str, ProteinFitnessTask]:
    root = Path(config_dir)
    out: dict[str, ProteinFitnessTask] = {}
    for path in sorted(root.glob("*.json")):
        task = load_task_config(path)
        out[task.name] = task
    return out


def one_hot_encode_sequences(
    sequences: Sequence[str],
    *,
    alphabet: str = CANONICAL_AA,
    sequence_length: int | None = None,
    channel_first: bool = True,
) -> Tensor:
    cleaned = [validate_sequence(seq, alphabet=alphabet) for seq in sequences]
    if sequence_length is None:
        lengths = {len(seq) for seq in cleaned}
        if len(lengths) != 1:
            raise ValueError("sequence_length is required when generated sequences have varying lengths")
        sequence_length = int(next(iter(lengths)))
    aa_to_idx = {aa: idx for idx, aa in enumerate(str(alphabet))}
    encoded = torch.zeros((len(cleaned), int(sequence_length), len(aa_to_idx)), dtype=torch.float)
    for row, seq in enumerate(cleaned):
        if len(seq) != int(sequence_length):
            raise ValueError(f"sequence length {len(seq)} does not match expected {int(sequence_length)}")
        for pos, aa in enumerate(seq):
            encoded[row, pos, aa_to_idx[aa]] = 1.0
    if channel_first:
        encoded = encoded.transpose(1, 2).contiguous()
    return encoded


def tokenize_ggs_sequences(
    sequences: Sequence[str],
    *,
    alphabet: str = GGS_ALPHABET,
    sequence_length: int | None = None,
) -> Tensor:
    """Tokenize protein sequences for the official Kirjner/GGS BaseCNN."""
    aa = str(alphabet)
    aa_to_idx = {residue: idx for idx, residue in enumerate(aa)}
    if len(aa_to_idx) != len(aa):
        raise ValueError("GGS oracle alphabet contains duplicate residues")
    cleaned: list[str] = []
    for seq in sequences:
        clean = str(seq).upper().strip()
        invalid = sorted({residue for residue in clean if residue not in aa_to_idx})
        if invalid:
            raise ValueError(f"GGS oracle sequence contains unknown residues: {''.join(invalid)}")
        cleaned.append(clean)
    if sequence_length is None:
        lengths = {len(seq) for seq in cleaned}
        if len(lengths) != 1:
            raise ValueError("sequence_length is required when generated sequences have varying lengths")
        sequence_length = int(next(iter(lengths)))
    tokens = torch.empty((len(cleaned), int(sequence_length)), dtype=torch.long)
    for row, seq in enumerate(cleaned):
        if len(seq) != int(sequence_length):
            raise ValueError(f"sequence length {len(seq)} does not match expected {int(sequence_length)}")
        for pos, residue in enumerate(seq):
            tokens[row, pos] = aa_to_idx[residue]
    return tokens


class _GGSLengthMaxPool1D(nn.Module):
    """Official GGS embedding block: linear/ReLU per position, then max pool."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.layer = nn.Linear(int(input_size), int(output_size))

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.layer(x))
        return torch.max(h, dim=1)[0]


class _GGSCNNPredictor(nn.Module):
    def __init__(
        self,
        *,
        n_tokens: int,
        hidden_size: int,
        kernel_size: int,
        embedding_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_tokens = int(n_tokens)
        self.encoder = nn.Conv1d(int(n_tokens), int(hidden_size), int(kernel_size))
        self.embedding = _GGSLengthMaxPool1D(int(hidden_size), int(embedding_size))
        self.dropout = nn.Dropout(float(dropout))
        self.decoder = nn.Linear(int(embedding_size), 1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"GGS CNN expects integer tokens shaped [batch, seq_len], got {tuple(x.shape)}")
        h = F.one_hot(x.long(), num_classes=self.n_tokens).permute(0, 2, 1).float()
        h = self.encoder(h).permute(0, 2, 1)
        h = self.dropout(h)
        h = self.embedding(h)
        return self.decoder(h).squeeze(1)


class _GGSCNNOracleModule(nn.Module):
    def __init__(
        self,
        *,
        n_tokens: int,
        hidden_size: int,
        kernel_size: int,
        embedding_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.predictor = _GGSCNNPredictor(
            n_tokens=n_tokens,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            embedding_size=embedding_size,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.predictor(x)


def _load_ggs_cnn_checkpoint(path: Path, *, map_location: torch.device) -> nn.Module:
    try:
        blob = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=map_location)
    if not isinstance(blob, Mapping) or "state_dict" not in blob:
        raise ValueError("GGS CNN checkpoint must contain a Lightning state_dict")
    state = blob["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("GGS CNN checkpoint state_dict must be a mapping")
    stripped_state = {
        key.removeprefix("predictor."): value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("predictor.")
    }
    if len(stripped_state) != len(state):
        unexpected = sorted(str(key) for key in state if not (isinstance(key, str) and key.startswith("predictor.")))
        raise ValueError(f"GGS CNN checkpoint contains unexpected non-predictor keys: {unexpected}")
    encoder_weight = stripped_state["encoder.weight"]
    embedding_weight = stripped_state["embedding.layer.weight"]
    model = _GGSCNNOracleModule(
        n_tokens=int(encoder_weight.shape[1]),
        hidden_size=int(encoder_weight.shape[0]),
        kernel_size=int(encoder_weight.shape[2]),
        embedding_size=int(embedding_weight.shape[0]),
    )
    try:
        model.predictor.load_state_dict(stripped_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Strict GGS CNN checkpoint load failed for {path}: {exc}") from exc
    return model.to(map_location)


class KirjnerCNNFitnessOracle:
    """Evaluation-only wrapper around the pretrained Kirjner et al. 1D CNN oracle."""

    evaluation_only = True

    def __init__(
        self,
        *,
        model: nn.Module | Callable[[Tensor], Tensor] | None = None,
        model_path: str | Path | None = None,
        stats: OracleNormalizationStats,
        sequence_length: int | None = None,
        alphabet: str = GGS_ALPHABET,
        device: str | torch.device = "cpu",
        channel_first: bool = True,
    ) -> None:
        if model is None and model_path is None:
            raise ValueError("KirjnerCNNFitnessOracle requires model or model_path")
        self.stats = stats
        self.sequence_length = None if sequence_length is None else int(sequence_length)
        self.alphabet = str(alphabet)
        self.device = torch.device(device)
        self.channel_first = bool(channel_first)
        self._uses_ggs_checkpoint_tokens = model is None and model_path is not None
        self.model = model if model is not None else self._load_model(Path(model_path or ""))
        if isinstance(self.model, nn.Module):
            self.model.to(self.device)
            self.model.eval()

    def _load_model(self, path: Path) -> nn.Module:
        if not path.exists():
            raise FileNotFoundError(f"Kirjner CNN oracle model was not found: {path}")
        try:
            model = torch.jit.load(str(path), map_location=self.device)
        except RuntimeError as exc:
            try:
                model = _load_ggs_cnn_checkpoint(path, map_location=self.device)
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Expected either a TorchScript export or the unmodified Kirjner/GGS "
                    f"CNN Lightning checkpoint. Checkpoint load error: {fallback_exc}"
                ) from fallback_exc
        model.eval()
        return model

    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        if not sequences:
            return np.zeros((0,), dtype=np.float64)
        if self._uses_ggs_checkpoint_tokens:
            x = tokenize_ggs_sequences(
                sequences,
                alphabet=self.alphabet,
                sequence_length=self.sequence_length,
            ).to(self.device)
        else:
            x = one_hot_encode_sequences(
                sequences,
                alphabet=self.alphabet,
                sequence_length=self.sequence_length,
                channel_first=self.channel_first,
            ).to(self.device)
        with torch.no_grad():
            raw = self.model(x)  # type: ignore[misc, operator]
        return torch.as_tensor(raw, dtype=torch.float32).detach().cpu().view(-1).numpy()

    def normalized_fitness(self, sequences: Sequence[str]) -> np.ndarray:
        raw = self.predict(sequences)
        denom = float(self.stats.y_max) - float(self.stats.y_min)
        return (raw - float(self.stats.y_min)) / denom


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return int(previous[-1])


def diversity(sequences: Sequence[str]) -> float:
    cleaned = [validate_sequence(seq) for seq in sequences]
    if len(cleaned) < 2:
        return 0.0
    distances = [levenshtein_distance(a, b) for a, b in combinations(cleaned, 2)]
    return float(np.median(np.asarray(distances, dtype=np.float64)))


def novelty(sequences: Sequence[str], training_sequences: Sequence[str]) -> float:
    cleaned = [validate_sequence(seq) for seq in sequences]
    training = [validate_sequence(seq) for seq in training_sequences]
    if not training:
        raise ValueError("training_sequences are required to compute novelty")
    nearest: list[int] = []
    for seq in cleaned:
        distances = [levenshtein_distance(seq, train_seq) for train_seq in training if train_seq != seq]
        nearest.append(min(distances) if distances else 0)
    return float(np.median(np.asarray(nearest, dtype=np.float64)))


def evaluate_generated_sequences(
    sequences: Sequence[str],
    *,
    oracle: KirjnerCNNFitnessOracle,
    training_sequences: Sequence[str] | None = None,
) -> ProteinFitnessEvaluation:
    if not sequences:
        raise ValueError("At least one generated sequence is required for evaluation")
    raw = oracle.predict(sequences)
    normalized = (raw - float(oracle.stats.y_min)) / (float(oracle.stats.y_max) - float(oracle.stats.y_min))
    novelty_value = None if training_sequences is None else novelty(sequences, training_sequences)
    return ProteinFitnessEvaluation(
        median_normalized_fitness=float(np.median(normalized)),
        diversity=diversity(sequences),
        novelty=novelty_value,
        raw_fitness=[float(v) for v in raw.tolist()],
        normalized_fitness=[float(v) for v in normalized.tolist()],
        num_sequences=len(sequences),
    )


_FORBIDDEN_TRAINING_KEYS = {
    "oracle",
    "oracle_model_path",
    "oracle_path",
    "full_dms",
    "full_dms_labels",
    "full_dms_path",
    "full_dms_y_min",
    "full_dms_y_max",
    "y_min",
    "y_max",
}


def assert_no_oracle_access(config: Mapping[str, Any], *, allow_oracle_access: bool = False) -> None:
    """Reject training configs that contain evaluation-oracle or full-DMS fields."""
    if allow_oracle_access:
        return

    def walk(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_str = str(key)
                if key_str in _FORBIDDEN_TRAINING_KEYS:
                    location = ".".join((*path, key_str))
                    raise ValueError(
                        f"Training config may not access evaluation-only oracle/full-DMS field '{location}'"
                    )
                walk(child, (*path, key_str))
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                walk(child, (*path, str(idx)))

    walk(config)


def sequences_from_jsonl(path: str | Path, *, alphabet: str = CANONICAL_AA) -> list[str]:
    """Load generated sequences from a JSONL file written by search or evaluation scripts."""
    out: list[str] = []
    aa = str(alphabet)
    for line_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "sequence" in row:
            out.append(validate_sequence(str(row["sequence"]), alphabet=aa))
            continue
        if "decoded" in row:
            out.append(validate_sequence(str(row["decoded"]), alphabet=aa))
            continue
        if "seq" in row and isinstance(row["seq"], list):
            try:
                out.append("".join(aa[int(token)] for token in row["seq"]))
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid token sequence at {path}:{line_no}") from exc
            continue
        raise ValueError(f"{path}:{line_no} must contain 'sequence', 'decoded', or list field 'seq'")
    return out


__all__ = [
    "KirjnerCNNFitnessOracle",
    "GGS_ALPHABET",
    "OracleNormalizationStats",
    "ProteinFitnessEvaluation",
    "ProteinFitnessTask",
    "assert_no_oracle_access",
    "diversity",
    "evaluate_generated_sequences",
    "levenshtein_distance",
    "load_all_task_configs",
    "load_task_config",
    "novelty",
    "one_hot_encode_sequences",
    "sequences_from_jsonl",
    "tokenize_ggs_sequences",
]
