from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn


class EMAWeightAveraging(WeightAveraging):
    def __init__(self, decay: float = 0.999):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay))
