from lightning.pytorch.cli import LightningCLI

from tiny_recursive_model.lit_trm import LitTRM
from tiny_recursive_model.sudoku import SudokuDataModule

# TODO:
# - automatic lr
# - reduce duplication in train_step and forward
# - With 10% probability (halt_exploration_prob: 0.1), the model is forced to continue for a random number of steps (2 to 16)
#   - min_halt_steps = (rand() < 0.1) * randint(2, 16)
#   - halted = halted & (steps >= min_halt_steps)
#   - This prevents early halting for exploration purposes
# - EMA
# - vocab size of 11 seems out by 1?
# - stablemax_cross_entropy loss function
# - validation step is dividing by samples...
#   - get to the bottom of sum vs mean over batch for loss, especially in context of early stopping
#   - ok, so is divided by batch size, just before backward() and before logging... must be do with distributed training?
#     - just use standard MEAN
#     - must then divide by K for microbatching
#     - use automatic batch size search
#   - early halting issue: halted puzzles get replaced with fresh puzzles, so batch size is constant
#     - each slot has a counter, so stops at 16 steps
#     - i could move the N_supervision steps to the outer loop, around each epoch, and keep track of the halted ones
# - log both losses, check they are in line... also double check paper how it blends

def main():
    LightningCLI(LitTRM, SudokuDataModule)


if __name__ == "__main__":
    main()
