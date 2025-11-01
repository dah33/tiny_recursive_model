import lightning as L
import torch
from einops import rearrange, reduce
from torch import Tensor
from torch.nn import functional as F

# from tiny_recursive_model.lit_ema import EMA_Callback
from tiny_recursive_model.paper.model import TinyRecursiveModel


def latent_recursion(net, x, y, z, n=6):  # all are (B, L, D)
    # the network learns to refine the latents if input is passed in, otherwise it refines the output
    for _ in range(n):  # latent reasoning
        z = net(y + z + x)
    y = net(y + z)  # refine output answer
    return y, z


def deep_recursion(net, output_head, Q_head, x, y, z, n=6, T=3):
    # recursing T−1 times to improve y and z (no gradients needed)
    with torch.no_grad():
        for _ in range(T - 1):
            y, z = latent_recursion(net, x, y, z, n)
    # recursing T−1 times to improve y and z (no gradients needed)
    y, z = latent_recursion(net, x, y, z, n)
    return (y.detach(), z.detach()), output_head(y), Q_head(y)


def softmax_cross_entropy(token_logits: Tensor, target_tokens: Tensor) -> Tensor:
    # F.cross_entropy expects (B, D, ...)
    token_logits = rearrange(token_logits, "B L D -> B D L")
    per_token_loss = F.cross_entropy(
        token_logits, target_tokens, reduction="none"
    )  # (B, L)
    return reduce(per_token_loss, "B ... -> B", "mean").sum()  # scalar


def binary_cross_entropy(
    halt_logits: Tensor, token_logits: Tensor, target_tokens: Tensor
) -> Tensor:
    # is the whole sequence correct?
    pred_tokens = token_logits.argmax(dim=-1)
    is_seq_correct = (pred_tokens == target_tokens).all(dim=-1)
    return F.binary_cross_entropy_with_logits(
        halt_logits, is_seq_correct.to(halt_logits.dtype), reduction="sum"
    )


class LitTRM(L.LightningModule):
    def __init__(
        self,
        *,
        # Model
        vocab_size: int = 11,
        D: int = 512,
        L: int = 81,
        scratch_slots: int = 16,
        n_layers: int = 2,
        expansion_factor: float = 4.0,
        rms_norm_eps: float = 1e-5,
        # Recursion
        T: int = 3,
        n: int = 6,
        halt_prob_threshold: float = 0.5,
        N_supervision: int = 16,
        # Training
        batch_split: int = 1,
    ):
        super().__init__()
        self.automatic_optimization = False

        # Save for checkpointing
        self.save_hyperparameters()

        # Hyperparameters used in training and forward
        self.halt_prob_threshold = halt_prob_threshold
        self.N_supervision = N_supervision
        self.batch_split = batch_split

        self.model = TinyRecursiveModel(
            vocab_size=vocab_size,
            D=D,
            L=L,
            scratch_slots=scratch_slots,
            n_layers=n_layers,
            expansion_factor=expansion_factor,
            rms_norm_eps=rms_norm_eps,
            n=n,
            T=T,
        )


    def training_step(self, batch, batch_idx) -> Tensor:
        # Manual optimization: https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        # Similar to TBPTT: https://lightning.ai/docs/pytorch/stable/common/tbptt.html

        opt, sch = self.optimizers(), self.lr_schedulers()

        # Reuse the same batch for N_supervision steps (batch may shrink due to halting)
        x_input_batch, y_true_batch, puzzle_ids = batch.values()
        y_batch, z_batch = None, None  # initial hiddens created by model
        halt_batch = torch.zeros_like(puzzle_ids, dtype=torch.bool)
        initial_batch_size = x_input_batch.shape[0]
        samples = 0

        for _ in range(self.N_supervision):

            # Accumulate gradients over K microbatches
            K = self.batch_split
            y_acc, z_acc, halt_acc = [], [], []
            loss_acc = 0.0
            for x_input, y, z, y_true, halt in zip(
                x_input_batch.chunk(K),
                y_batch.chunk(K) if y_batch is not None else [None] * K,
                z_batch.chunk(K) if z_batch is not None else [None] * K,
                y_true_batch.chunk(K),
                halt_batch.chunk(K),
            ):
                # Deep recursion step
                (y, z), y_hat, q_hat = self.model(x_input, y, z)
                pred_loss = softmax_cross_entropy(y_hat, y_true)
                halt_loss = binary_cross_entropy(q_hat, y_hat, y_true)
                loss = pred_loss + halt_loss  # sum loss, hence no divide by K
                self.manual_backward(loss)

                # Should we halt supervision for each puzzle?
                halt = q_hat.detach() >= self.halt_prob_threshold

                # Accumulate microbatch results
                y_acc.append(y)
                z_acc.append(z)
                halt_acc.append(halt)
                loss_acc += loss.item()
                samples += x_input.shape[0]

            # Optimize based on accumulated gradients for full batch
            opt.step()
            opt.zero_grad()
            sch.step()

            # Combine microbatch results
            y_batch = torch.cat(y_acc, dim=0)
            z_batch = torch.cat(z_acc, dim=0)
            halt_batch = torch.cat(halt_acc, dim=0)

            # Early stopping if all puzzles have halted
            if halt_batch.all():
                break

            # Keep only active puzzles for next supervision step
            active = ~halt_batch
            x_input_batch = x_input_batch[active]
            y_batch = y_batch[active]
            z_batch = z_batch[active]
            y_true_batch = y_true_batch[active]

        # Log batch metrics
        avg_sup_steps = samples / initial_batch_size * self.N_supervision
        self.log("loss", loss_acc, on_step=True, prog_bar=True)
        self.log("sup", avg_sup_steps, on_step=True, prog_bar=True)

    def forward(self, batch, compute_loss=False):
        x_input, y_true, puzzle_ids = batch.values()
        y, z = None, None
        halt = torch.zeros_like(puzzle_ids, dtype=torch.bool)
        total_loss = 0.0
        samples = 0

        # No microbatches required
        for _ in range(self.N_supervision):
            (y, z), y_hat, q_hat = self.model(x_input, y, z)

            if compute_loss:
                pred_loss = softmax_cross_entropy(y_hat, y_true)
                halt_loss = binary_cross_entropy(q_hat, y_hat, y_true)
                loss = pred_loss + halt_loss
                total_loss += loss.item()
                samples += x_input.shape[0]

            halt = q_hat >= self.halt_prob_threshold
            if halt.all():
                break

            active = ~halt
            x_input = x_input[active]
            y = y[active]
            z = z[active]
            if compute_loss:
                y_true = y_true[active]

        return y_hat, total_loss, samples

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat, _, _ = self.forward(batch, compute_loss=False)
        return y_hat.argmax(dim=-1)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _y_hat, total_loss, samples = self.forward(batch, compute_loss=True)
        avg_loss = total_loss / samples if samples > 0 else 0.0
        # TODO: why do we need on_step=False if validation_step only called on epoch end?
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        return avg_loss
