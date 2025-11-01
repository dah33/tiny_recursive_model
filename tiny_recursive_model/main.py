import sys

from lightning.pytorch.cli import LightningCLI

from tiny_recursive_model.lit_trm import LitTRM
from tiny_recursive_model.sudoku import SudokuDataModule

# TODO:
# - validation step
#   - validation set is distinct puzzle id, so take last n*1000 puzzles from train as val?#   - train test split differs from paper currently, as 90-10 (paper has 1m in train)
# - reduce duplication in train_step and forward
# - EMA

def main():
    LightningCLI(LitTRM, SudokuDataModule)


if __name__ == "__main__":
    main()
