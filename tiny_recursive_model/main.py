import sys

from lightning.pytorch.cli import LightningCLI

from tiny_recursive_model.lit_trm import LitTRM
from tiny_recursive_model.sudoku import SudokuDataModule

# TODO:
# - new repo
# - validation step
#   - shorten epoch , to make easier to see validation
#   - validation set is distinct puzzle id, so take last n*1000 puzzles from train as val?
# - max_epochs=60... perhaps a short run and a long run
# - seed=42
# - EMA


def main():
    LightningCLI(LitTRM, SudokuDataModule)


if __name__ == "__main__":
    sys.argv = [
        "main.py",
        "fit",
        # "--trainer.fast_dev_run=true",
        "--trainer.precision=bf16-mixed",
        "--data.batch_size=768",
        "--model.batch_split=4",
        "--model.n=2",
        "--model.T=2",
        "--model.N_supervision=1",
    ]
    main()
