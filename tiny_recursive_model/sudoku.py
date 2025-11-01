from pathlib import Path

import lightning as L
import numpy as np
import torch
from metadata import load_metadata
from torch.utils.data import DataLoader, Dataset, random_split

# Dataset Metadata:
# - seq_len=81,
# - vocab_size=10 + 1,  # - PAD + "0" ... "9"
# - pad_id=0,
# - ignore_label_id=0,
# - blank_identifier_id=0,
# - num_puzzle_identifiers=1,
# - total_groups=len(results["group_indices"]) - 1,
# - mean_puzzle_examples=1,
# - total_puzzles=len(results["group_indices"]) - 1,
# - sets=["all"]

# Architecture parameters (from trm.yaml), with sudoku command line overrides (from README.md; prefixed with "arch."):
# - loss_type: stablemax_cross_entropy
# - halt_exploration_prob: 0.1
# - halt_max_steps: 16
# - H_cycles: 3
# - L_cycles: 6
# - H_layers: 0 # not used
# - L_layers: 2
# - hidden_size: 512
# - puzzle_emb_ndim: ${.hidden_size}
# - puzzle_emb_len: 16 # if non-zero, its specified to this value
# - num_heads: 8  # min(2, hidden_size // 64)
# - expansion: 4
# - pos_encodings: none
# - forward_dtype: bfloat16
# - mlp_t: True # use mlp on L instead of transformer
# - no_ACT_continue: True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense
#
# Training parameters (from cfg_pretrain.yaml), with sudoku command line overrides (from README.md):
# # Hyperparams - Training
# - global_batch_size: 768
# - epochs=50000
# - eval_interval=5000
# - checkpoint_every_eval: True
# - lr=1e-4
# - lr_min_ratio: 1.0
# - lr_warmup_steps: 2000
# # Standard hyperparameter settings for LM, as used in Llama
# - beta1: 0.9
# - beta2: 0.95
# - weight_decay=1.0
# - puzzle_emb_weight_decay=1.0
# - puzzle_emb_lr=1e-4
# - seed: 0
# - min_eval_interval: 0 # when to start the eval
# - ema=True
# - ema_rate: 0.999 # EMA-rate
# - evaluators="[]"
# - freeze_weights: False # If True, freeze weights and only learn the embeddings
# - data_paths="[data/sudoku-extreme-1k-aug-1000]"
#
# Defaults from TinyRecursiveReasoningModel_ACTV1Config (if not specified above):
# - puzzle_emb_ndim: int = 0
# - rms_norm_eps: float = 1e-5
# - rope_theta: float = 10000.0 # not using positional encodings
# - forward_dtype: str = "bfloat16"


class SudokuDataset(Dataset):
    """PyTorch Dataset for Sudoku puzzles."""

    def __init__(self, data_path: Path):
        super().__init__()

        self.inputs = torch.from_numpy(
            np.load(data_path / "all__inputs.npy").astype(np.int32)
        )
        self.labels = torch.from_numpy(
            np.load(data_path / "all__labels.npy").astype(np.int64)
        )
        self.puzzle_identifiers = torch.from_numpy(
            np.load(data_path / "all__puzzle_identifiers.npy").astype(np.int32)
        )
        assert len(self.inputs) == len(self.labels) == len(self.puzzle_identifiers)

        self.metadata = load_metadata(data_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """Return a single example."""
        return {
            "input": self.inputs[idx],
            "label": self.labels[idx],
            "puzzle_id": self.puzzle_identifiers[idx],
        }


class SudokuDataModule(L.LightningDataModule):
    """Lightning DataModule for Sudoku dataset."""

    def __init__(
        self,
        root_dir: str = "data/sudoku-extreme-1k-aug-1000",
        batch_size: int = 768,
        num_workers: int = 4,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sudoku_train = None
        self.sudoku_val = None

    def setup(self, stage: str | None = None):
        """Setup datasets for training and validation.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        sudoku_full = SudokuDataset(self.root_dir / "train")
        self.sudoku_train, self.sudoku_val = random_split(sudoku_full, [0.9, 0.1])
        self.sudoku_test = SudokuDataset(self.root_dir / "test")

    def train_dataloader(self):
        return DataLoader(
            self.sudoku_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.sudoku_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.sudoku_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
