import lightning as L
import torch
from einops import rearrange, reduce
from torch import Tensor
from torch.nn import functional as F

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
    flat_logits = rearrange(token_logits, "B L D -> (B L) D")
    flat_target = rearrange(target_tokens, "B L -> (B L)")
    return F.cross_entropy(flat_logits, flat_target)  # mean over flattened B x L


def binary_cross_entropy(
    halt_logits: Tensor, token_logits: Tensor, target_tokens: Tensor
) -> Tensor:
    # 1 if the entire predicted sequence matches the target sequence
    is_seq_correct = (token_logits.argmax(dim=-1) == target_tokens).all(dim=-1)
    target = is_seq_correct.to(dtype=halt_logits.dtype)
    return F.binary_cross_entropy_with_logits(halt_logits, target)  # mean over batch


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
        final_loss, final_halt_loss = 0.0, 0.0
        batch_size = x_input_batch.shape[0]
        sample_count = 0

        for supervision_step in range(self.N_supervision):

            # Accumulate gradients over K microbatches
            K = self.batch_split
            y_acc, z_acc, halt_acc = [], [], []
            loss_acc, halt_loss_acc = 0.0, 0.0
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
                loss = pred_loss + halt_loss
                self.manual_backward(loss / K)  # accumulate over K microbatches

                # Should we halt supervision for each puzzle?
                halt = q_hat.detach() >= self.halt_prob_threshold

                # Accumulate microbatch results
                y_acc.append(y)
                z_acc.append(z)
                halt_acc.append(halt)
                loss_acc += loss.item() / K
                halt_loss_acc += halt_loss.item() / K
                sample_count += x_input.shape[0]

            # Optimize based on accumulated gradients for full batch
            opt.step()
            opt.zero_grad()
            sch.step()

            # Combine microbatch results
            y_batch = torch.cat(y_acc, dim=0)
            z_batch = torch.cat(z_acc, dim=0)
            halt_batch = torch.cat(halt_acc, dim=0)

            # Accumulate losses if at final supervision step
            is_final_step = halt_batch | (supervision_step == self.N_supervision - 1)
            final_loss += loss_acc * is_final_step.sum() / batch_size
            final_halt_loss += halt_loss_acc * is_final_step.sum() / batch_size

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
        avg_sup_steps = sample_count / batch_size * self.N_supervision
        self.log("loss", final_loss, on_step=True, prog_bar=True)
        self.log("halt_loss", final_halt_loss, on_step=True, prog_bar=True)
        self.log("avg_sup", avg_sup_steps, on_step=True, prog_bar=True)

    def forward(self, batch):
        x_input, y_true, puzzle_ids = batch.values()
        y, z = None, None
        halt = torch.zeros_like(puzzle_ids, dtype=torch.bool)
        final_loss, final_halt_loss = 0.0, 0.0
        batch_size = x_input.shape[0]
        sample_count = 0

        # No microbatches required
        for supervision_step in range(self.N_supervision):
            (y, z), y_hat, q_hat = self.model(x_input, y, z)

            pred_loss = softmax_cross_entropy(y_hat, y_true)
            halt_loss = binary_cross_entropy(q_hat, y_hat, y_true)
            loss = (pred_loss + halt_loss).item()
            sample_count += x_input.shape[0]

            halt = q_hat >= self.halt_prob_threshold

            # Accumulate losses if at final supervision step
            is_final_step = halt | (supervision_step == self.N_supervision - 1)
            final_loss += loss * is_final_step.sum() / batch_size
            final_halt_loss += halt_loss * is_final_step.sum() / batch_size

            if halt.all():
                break

            active = ~halt
            x_input = x_input[active]
            y = y[active]
            z = z[active]
            y_true = y_true[active]

        avg_sup_steps = sample_count / batch_size * self.N_supervision
        return y_hat, (final_loss, final_halt_loss, avg_sup_steps)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat, _ = self.forward(batch)
        return y_hat.argmax(dim=-1)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, (loss, halt_loss, avg_sup_steps) = self.forward(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("halt_loss", halt_loss)
        self.log("avg_sup_steps", avg_sup_steps)
