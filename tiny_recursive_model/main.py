import sys

from lightning.pytorch.cli import LightningCLI

from tiny_recursive_model.lit_trm import LitTRM
from tiny_recursive_model.sudoku import SudokuDataModule

# TODO:
# - reduce duplication in train_step and forward
# - With 10% probability (halt_exploration_prob: 0.1), the model is forced to continue for a random number of steps (2 to 16)
#   - min_halt_steps = (rand() < 0.1) * randint(2, 16)
#   - halted = halted & (steps >= min_halt_steps)
#   - This prevents early halting for exploration purposes
# - EMA
# - vocab size of 11 seems out by 1?
# - stablemax_cross_entropy loss function
# - save_hyperparameters or modern equivalent -> resume from checkpoint
# - validation step is dividing by samples... get to the bottom of sum vs mean over batch for loss
# - log both losses, check they are in line... also double check paper how it blends

def main():
    LightningCLI(LitTRM, SudokuDataModule)


if __name__ == "__main__":
    main()
