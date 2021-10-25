
from utils import *
from pycox.models.loss import NLLLogistiHazardLoss


class WassLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, psi: Tensor) -> Tensor:
        a, b = sepr_repr(psi)
        self.psi0 = a
        self.psi1 = b
        return SinkhornDistance(eps=0.001, max_iter=100, reduction=None)(a, b)


class _Loss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction


def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(
        f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")


def nll_pmf(phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = 'mean', epsilon: float = 1e-7) -> Tensor:
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`." +
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`," +
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    idx_durations = idx_durations.view(-1, 1)
    phi = utils.pad_col(phi)
    gamma = phi.max(1)[0]
    cumsum = phi.sub(gamma.view(-1, 1)).exp().cumsum(1)
    sum_ = cumsum[:, -1]
    part1 = phi.gather(1, idx_durations).view(-1).sub(gamma).mul(events)
    part2 = - sum_.relu().add(epsilon).log()
    part3 = sum_.sub(cumsum.gather(1, idx_durations).view(-1)
                     ).relu().add(epsilon).log().mul(1. - events)
    # need relu() in part3 (and possibly part2) because cumsum on gpu has some bugs and we risk getting negative numbers.
    loss = - part1.add(part2).add(part3)
    return _reduction(loss, reduction)


def new_loss(phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = 'mean', epsilon: float = 1e-7) -> Tensor:
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`." +
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`," +
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    idx_durations = idx_durations.view(-1, 1)
    phi = utils.pad_col(phi)
    cumsum = phi.exp().cumsum(1)
    sum_ = cumsum[:, -1]
    norm = 1 + sum_.sum()
    part1 = sum_.add(epsilon).log().div(norm)  # relu()
    part2 = sum_.sub(cumsum.gather(1, idx_durations).view(-1)
                     ).add(epsilon).div(norm).log().mul(1. - events)  # relu()
    loss = - part1.add(part2)
    return _reduction(loss, reduction)


class NLLPMFLoss(_Loss):
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
        return nll_pmf(phi, idx_durations, events, self.reduction)


class NewLoss(_Loss):
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
        return new_loss(phi, idx_durations, events, self.reduction)


class Loss(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """

    def __init__(self,  alpha, beta=1):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.loss_surv = NLLPMFLoss()  # NewLoss()# NLLLogistiHazardLoss()
        self.loss_wass = WassLoss()  # IPM

    def forward(self, phi, psi_t, idx_durations, events):
        events = events.clone().detach().float()  # torch.tensor(events).float()
        # Survival loss.
        loss_surv = self.loss_surv(phi, idx_durations, events)
        loss_wass = self.loss_wass(psi_t)  # Wasserstein Loss

        return self.beta*loss_surv + self.alpha / phi.shape[0] * loss_wass
