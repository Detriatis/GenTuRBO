"""ESM-2 protein sequence featurization."""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

import torch

from adapters.protein.validation import validate_sequence

ModelLoader = Callable[[str], tuple[Any, Any]]


def infer_esm2_repr_layer(model_name: str) -> int:
    """Infer the final representation layer from an ESM model name."""
    match = re.search(r"_t(\d+)_", str(model_name))
    if match is None:
        raise ValueError(f"Could not infer ESM repr layer from model_name='{model_name}'")
    return int(match.group(1))


class ESM2Featurizer:
    """Mean-pooled ESM-2 residue embeddings with BOS/EOS excluded."""

    def __init__(
        self,
        *,
        model_name: str = "esm2_t30_150M_UR50D",
        repr_layer: int | None = None,
        device: str | torch.device = "cpu",
        batch_size: int = 16,
        model_loader: Optional[ModelLoader] = None,
    ) -> None:
        self.model_name = str(model_name)
        self.repr_layer = int(repr_layer) if repr_layer is not None else infer_esm2_repr_layer(self.model_name)
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self._model_loader = model_loader
        self._model: Any = None
        self._alphabet: Any = None
        self._batch_converter: Any = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is not None and self._alphabet is not None and self._batch_converter is not None:
            return self._model, self._alphabet, self._batch_converter

        if self._model_loader is None:
            try:
                import esm
            except ImportError as exc:
                raise ImportError("fair-esm is required to use ESM2Featurizer") from exc
            if self.model_name != "esm2_t33_650M_UR50D":
                raise ValueError("This reproduction release pins only ESM2 t33 650M UR50D weights")
            from genturbo.model_weights import prepare_weights
            prepare_weights(download=False, names=("esm2", "esm2-regression"))
            loader = getattr(esm.pretrained, self.model_name)
            model, alphabet = loader()
        else:
            model, alphabet = self._model_loader(self.model_name)

        if hasattr(model, "eval"):
            model = model.eval()
        if hasattr(model, "to"):
            model = model.to(self.device)
        batch_converter = alphabet.get_batch_converter()

        self._model = model
        self._alphabet = alphabet
        self._batch_converter = batch_converter
        return model, alphabet, batch_converter

    def featurize(self, seqs: list[str]) -> torch.Tensor:
        """Return a tensor of shape ``[len(seqs), embedding_dim]``."""
        if not seqs:
            return torch.empty((0, 0), dtype=torch.float)
        clean = [validate_sequence(seq) for seq in seqs]
        model, alphabet, batch_converter = self._load()
        reps: list[torch.Tensor] = []
        padding_idx = getattr(alphabet, "padding_idx", None)

        with torch.no_grad():
            for start in range(0, len(clean), self.batch_size):
                chunk = clean[start : start + self.batch_size]
                data = [(f"seq_{start + idx}", seq) for idx, seq in enumerate(chunk)]
                _, _, tokens = batch_converter(data)
                tokens = tokens.to(self.device)
                results = model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
                token_reps = results["representations"][self.repr_layer]
                if padding_idx is None:
                    token_lens = torch.as_tensor([len(seq) + 2 for seq in chunk], device=tokens.device)
                else:
                    token_lens = (tokens != int(padding_idx)).sum(dim=1)
                for row_idx, seq in enumerate(chunk):
                    end = int(min(len(seq) + 1, int(token_lens[row_idx].item()) - 1))
                    if end <= 1:
                        raise ValueError("ESM tokenization produced no residue tokens")
                    reps.append(token_reps[row_idx, 1:end].mean(dim=0).detach().cpu())

        return torch.stack(reps, dim=0).to(dtype=torch.float)

    def __call__(self, seqs: list[str]) -> torch.Tensor:
        return self.featurize(seqs)


__all__ = ["ESM2Featurizer", "infer_esm2_repr_layer"]
