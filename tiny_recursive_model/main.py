from lightning.pytorch.cli import LightningCLI

from tiny_recursive_model.lit_trm import LitTRM
from tiny_recursive_model.sudoku import SudokuDataModule

# TODO:
# - I wonder if fast learning at start is due to vocab_size being 11 instead of 10?
#   - predictions should be 1-9 only, no point predicting pad or empty
# - resume training from checkpoint, now we have validation working
# - reduce duplication in train_step and forward
# - With 10% probability (halt_exploration_prob: 0.1), the model is forced to continue for a random number of steps (2 to 16)
#   - min_halt_steps = (rand() < 0.1) * randint(2, 16)
#   - halted = halted & (steps >= min_halt_steps)
#   - This prevents early halting for exploration purposes
# - EMA - can use the Weighted callback built into Lightning (soon)
# - stablemax_cross_entropy loss function
# - gpu utilisation: halted puzzles get replaced with fresh puzzles, so batch size is constant
# - double check paper how it blends the losses

def main():
    LightningCLI(LitTRM, SudokuDataModule)


if __name__ == "__main__":
    main()
