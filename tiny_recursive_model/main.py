import logging
import warnings

import torch
from lightning.pytorch.cli import LightningCLI

from tiny_recursive_model.lit_trm import LitTRM
from tiny_recursive_model.sudoku import SudokuDataModule


class TorchCompileCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--compile",
            type=bool,
            default=False,
            help="Enable torch.compile for the model (default: False, not recommended with microbatch_count > 1)",
        )

    def fit(self, model, **kwargs):
        if self.config.get("compile", False):
            # Compiled with expected warnings suppressed:
            warnings.filterwarnings(
                "ignore", message=".*does not support bfloat16 compilation natively.*"
            )
            warnings.filterwarnings("ignore", message=".*functools.lru_cache.*")
            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
            logging.getLogger("torch._inductor").setLevel(logging.ERROR)
            model = torch.compile(model)

        self.trainer.fit(model, **kwargs)


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
    TorchCompileCLI(LitTRM, SudokuDataModule)


if __name__ == "__main__":
    main()
