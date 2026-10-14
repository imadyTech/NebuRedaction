"""Model C: a tiny character-level CNN (NR-0004 §2 item 2, plan §4.3).

Plan §4.3 describes C as a PyTorch CPU model budgeted at 50-500 KB, whose role
is "上限参考" — an upper-bound reference rather than the expected default.

I first wrote here that it could not ship because G2 caps the model at 100 KB.
That was wrong, and measuring it is what showed so: at an embedding width of
16, 32 channels and kernels of 3, 5 and 7, the network is **10,023 parameters,
39.2 KB in float32** — well inside the limit even unquantised. So C is a
genuine candidate if it earns the place, and the report should say so if it
does rather than deferring to the plan's expectation.

What C is *for*, though, is a different question: **how much of the task is
the features and how much is the model?**

That makes C the control for the project's central design claim. Models A and B
read 31 hand-designed morphological features plus hashed n-grams; C reads raw
characters and is given no features at all. If C clearly beats them, the
features are leaving information on the table and NR-0005's .NET port is
carrying a hand-built pipeline it did not need. If C does not, the features are
where the work is — which is also the finding that makes a kilobyte-scale
linear model a sensible thing to ship at all.

**It sees the value only.** A and B get the key name, the template and the
scope; C gets the characters of the value. That is a deliberately unfair
comparison and it is the useful one: the key name is a hint about a value, so
giving C the value alone measures how much the *value* can say on its own,
which is precisely plan §3.3's hard case and NR-0002's finding 3 — the formats
with no structural marker.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datagen.unit import LABELS
from training.dataset import Dataset

from neburedaction.features import FEATURE_VERSION

#: Characters the model can see. Everything else maps to one "other" slot, so
#: the alphabet is fixed and small rather than learned from the corpus — a
#: vocabulary fitted to the training data would be another thing NR-0005 has to
#: reproduce exactly.
ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "+/=_-.:@;?&%$#!*~^|\\'\" ,()[]{}<>\n\t"
)

#: Index 0 is padding, 1 is "a character outside the alphabet".
PAD_INDEX = 0
OTHER_INDEX = 1
VOCABULARY_SIZE = len(ALPHABET) + 2

#: Characters kept from each value. A PEM block is about 1,700 characters and
#: the structural signal is in its first line, so truncating costs little and
#: bounds both memory and inference time.
MAX_LENGTH = 128


def encode(value: str, max_length: int = MAX_LENGTH) -> np.ndarray:
    """Encode a value as fixed-length character indices.

    Args:
        value: The attribute value.
        max_length: Length to pad or truncate to.

    Returns:
        An int64 array of ``max_length`` indices.
    """
    lookup = {character: index + 2 for index, character in enumerate(ALPHABET)}
    encoded = np.full(max_length, PAD_INDEX, dtype=np.int64)
    for position, character in enumerate(value[:max_length]):
        encoded[position] = lookup.get(character, OTHER_INDEX)
    return encoded


@dataclass(frozen=True, slots=True)
class CharCnnModel:
    """A trained character CNN.

    Attributes:
        state: The model's parameters, as numpy arrays.
        embedding_dim: Character embedding width.
        channels: Convolution channels.
        kernel_sizes: Convolution widths.
        max_length: Characters read per value.
        parameters: Total parameter count.
        feature_version: Recorded for comparability, though C uses no features.
        labels: Class order.
    """

    state: dict[str, np.ndarray]
    embedding_dim: int
    channels: int
    kernel_sizes: tuple[int, ...]
    max_length: int
    parameters: int
    feature_version: str
    labels: tuple[str, ...]

    @property
    def size_bytes(self) -> int:
        """Size as float32, which is what a PyTorch export would be.

        Not quantised to int8 like model A, and the asymmetry is deliberate
        rather than an oversight: A's quantisation cost is part of A's measured
        result, so quantising C too would compare A-after-a-cost against
        C-before-one. At 39.2 KB float32 C is inside the budget anyway, so
        nothing is being hidden by leaving it unquantised — if it were close to
        the limit the comparison would need redoing both ways.
        """
        return self.parameters * 4

    def to_dict(self) -> dict[str, Any]:
        """Serialise the metadata."""
        return {
            "kind": "char_cnn_float32",
            "embedding_dim": self.embedding_dim,
            "channels": self.channels,
            "kernel_sizes": list(self.kernel_sizes),
            "max_length": self.max_length,
            "vocabulary": VOCABULARY_SIZE,
            "parameters": self.parameters,
            "size_bytes": self.size_bytes,
            "size_kib": round(self.size_bytes / 1024, 1),
            "labels": list(self.labels),
            "reads": "value characters only, no engineered features",
            "role": (
                "upper-bound reference (plan §4.3), but inside G2's 100 KB limit at "
                "this size, so a candidate if it earns the place"
            ),
        }


def build_network(embedding_dim: int, channels: int, kernel_sizes: tuple[int, ...]) -> Any:
    """Build the network.

    Args:
        embedding_dim: Character embedding width.
        channels: Channels per convolution.
        kernel_sizes: Convolution widths, run in parallel and max-pooled.

    Returns:
        A ``torch.nn.Module``.

    Parallel convolutions of several widths with global max pooling: the widths
    correspond to character n-grams of those sizes, and max pooling makes the
    answer independent of *where* in the value a pattern occurs. A credential's
    marker can be a prefix (`ghp_`), an infix (`://user:pass@`) or a whole-string
    property, so position-independence is the right inductive bias and it is
    also what keeps the parameter count low enough to be comparable with A.
    """
    import torch
    from torch import nn

    class CharCnn(nn.Module):
        """A small parallel-convolution character classifier."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(VOCABULARY_SIZE, embedding_dim, padding_idx=PAD_INDEX)
            self.convolutions = nn.ModuleList(
                [
                    nn.Conv1d(embedding_dim, channels, kernel_size=size, padding=size // 2)
                    for size in kernel_sizes
                ]
            )
            self.output = nn.Linear(channels * len(kernel_sizes), len(LABELS))

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            """Map character indices to class logits."""
            embedded = self.embedding(tokens).transpose(1, 2)
            pooled = [
                torch.relu(convolution(embedded)).max(dim=2).values
                for convolution in self.convolutions
            ]
            return self.output(torch.cat(pooled, dim=1))

    return CharCnn()


def train(
    dataset: Dataset,
    validation: Dataset,
    embedding_dim: int = 16,
    channels: int = 32,
    kernel_sizes: tuple[int, ...] = (3, 5, 7),
    epochs: int = 3,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    threads: int = 10,
    seed: int = 20261014,
) -> tuple[CharCnnModel, dict[str, Any]]:
    """Fit model C.

    Args:
        dataset: The training split, used for its records and soft targets.
        validation: The validation split, scored each epoch.
        embedding_dim: Character embedding width.
        channels: Channels per convolution.
        kernel_sizes: Convolution widths.
        epochs: Passes over the training data.
        batch_size: Rows per step.
        learning_rate: Adam learning rate.
        threads: Torch thread cap, matching NR-0003 §2.2A item 1's ceiling of
            10 on this machine.
        seed: Seed, so a fit is reproducible.

    Returns:
        ``(model, diagnostics)``.

    Trained on the **soft targets directly**, by cross-entropy against a
    distribution. Unlike models A and B this needs no replication trick: a
    neural network's loss is written by hand anyway, so the distribution can be
    the target as it stands. The targets are the same ones A and B see, soft on
    distilled positives and one-hot elsewhere, so the comparison is about the
    model and not about what it was asked to predict.
    """
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(threads)

    encoded = np.stack([encode(record.value, MAX_LENGTH) for record in dataset.records])
    targets = dataset.targets
    val_encoded = np.stack([encode(record.value, MAX_LENGTH) for record in validation.records])
    val_hard = validation.hard

    network = build_network(embedding_dim, channels, kernel_sizes)
    optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)
    parameters = sum(parameter.numel() for parameter in network.parameters())

    tokens = torch.from_numpy(encoded)
    target_tensor = torch.from_numpy(targets).float()

    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        network.train()
        permutation = torch.randperm(tokens.shape[0], generator=generator)
        total_loss = 0.0
        batches = 0
        for start in range(0, tokens.shape[0], batch_size):
            selected = permutation[start : start + batch_size]
            optimiser.zero_grad()
            logits = network(tokens[selected])
            # Cross-entropy against a distribution, written out because
            # torch's functional cross_entropy takes class indices.
            loss = -(target_tensor[selected] * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss.backward()
            optimiser.step()
            total_loss += float(loss.item())
            batches += 1

        network.eval()
        with torch.no_grad():
            predictions = []
            for start in range(0, val_encoded.shape[0], 1024):
                chunk = torch.from_numpy(val_encoded[start : start + 1024])
                predictions.append(network(chunk).argmax(dim=1).numpy())
        predicted = np.concatenate(predictions)
        history.append(
            {
                "epoch": epoch,
                "mean_loss": round(total_loss / max(batches, 1), 5),
                "validation_class_accuracy": round(float((predicted == val_hard).mean()), 5),
            }
        )
        print(f"  epoch {epoch}: {history[-1]}", flush=True)

    fit_seconds = time.perf_counter() - started
    state = {name: tensor.detach().numpy() for name, tensor in network.state_dict().items()}

    model = CharCnnModel(
        state=state,
        embedding_dim=embedding_dim,
        channels=channels,
        kernel_sizes=kernel_sizes,
        max_length=MAX_LENGTH,
        parameters=parameters,
        feature_version=FEATURE_VERSION,
        labels=tuple(LABELS),
    )
    return model, {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "threads": threads,
        "seed": seed,
        "fit_seconds": round(fit_seconds, 1),
        "history": history,
        **model.to_dict(),
    }


def predict_proba(model: CharCnnModel, values: list[str]) -> np.ndarray:
    """Score values with a fitted model.

    Args:
        model: The fitted model.
        values: Attribute values.

    Returns:
        Class probabilities.
    """
    import torch

    network = build_network(model.embedding_dim, model.channels, model.kernel_sizes)
    network.load_state_dict({name: torch.from_numpy(array) for name, array in model.state.items()})
    network.eval()
    encoded = np.stack([encode(value, model.max_length) for value in values])
    with torch.no_grad():
        logits = network(torch.from_numpy(encoded))
        return torch.softmax(logits, dim=1).numpy().astype(np.float64)


def save(model: CharCnnModel, diagnostics: dict[str, Any], directory: Path) -> dict[str, Path]:
    """Write the model and its diagnostics.

    Args:
        model: The fitted model.
        diagnostics: Training diagnostics.
        directory: Where to write.

    Returns:
        The paths written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    arrays = directory / "model-c.npz"
    np.savez_compressed(arrays, **model.state)
    metadata = directory / "model-c.json"
    metadata.write_text(
        json.dumps({**model.to_dict(), "diagnostics": diagnostics}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"arrays": arrays, "metadata": metadata}
