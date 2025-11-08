from lightning.pytorch.cli import LightningCLI

from tiny_recursive_model.lit_trm import LitTRM
from tiny_recursive_model.sudoku import SudokuDataModule

# TODO:
# - activation checkpointing to save memory (reduce_memory parameter)
# - reduce duplication in train_step and forward
# - With 10% probability (halt_exploration_prob: 0.1), the model is forced to continue for a random number of steps (2 to 16)
#   - min_halt_steps = (rand() < 0.1) * randint(2, 16)
#   - halted = halted & (steps >= min_halt_steps)
#   - This prevents early halting for exploration purposes
# - higher learning rate
#   - EMA - can use the Weighted callback built into Lightning (soon)
#   - stablemax_cross_entropy loss function
# - paper has loss + 0.5 * (halt_loss  + continue_loss), so loss + 0.5 * halt_loss


def main():
    LightningCLI(LitTRM, SudokuDataModule)


if __name__ == "__main__":
    main()
