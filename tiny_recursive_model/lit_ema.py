import lightning as L
from ema_pytorch import EMA


class EMACallback(L.Callback):
    # Usage was:
    # EMACallback(
    #     decay=ema_decay_rate,
    #     update_every=1,
    #     apply_at_train_end=True,
    #     forward_method_names=("predict",),
    # )
    def __init__(
        self,
        decay: float = 0.999,
        update_every: int = 1,
        switch_ema_every: int = 10_000,
        apply_at_train_end: bool = True,
        forward_method_names: tuple[str, ...] = ("predict",),
    ):
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.switch_ema_every = switch_ema_every
        self.apply_at_train_end = apply_at_train_end
        self.forward_method_names = forward_method_names
        self.ema: EMA | None = None
        self._step = 0

    def on_fit_start(self, trainer: L.Trainer, pl_module: LitTRM):
        self.ema = EMA(
            pl_module.model,
            beta=self.decay,
            update_model_with_ema_every=self.switch_ema_every,
            forward_method_names=self.forward_method_names,
        )

    def on_after_backward(self, trainer: L.Trainer, pl_module: LitTRM):
        # update every optimizer step
        if self.ema is not None:
            self.ema.update()

    def on_train_end(self, trainer: L.Trainer, pl_module: LitTRM):
        if self.apply_at_train_end and self.ema is not None:
            self.ema.copy_params_from_ema_to_model()

    # helper for manual apply in scripts
    def apply_now(self):
        if self.ema is not None:
            self.ema.copy_params_from_ema_to_model()
