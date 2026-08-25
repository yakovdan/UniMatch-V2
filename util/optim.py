import logging
import math
import os

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    r"""Custom AdamW (Loshchilov & Hutter, "Decoupled Weight Decay Regularization").

    A plain per-parameter implementation of the same update rule as
    ``torch.optim.AdamW`` (single-tensor path), written to be easy to read and
    modify. Per step, for each parameter $\theta$ with gradient $g$:

    $$
    \begin{aligned}
    \theta &\leftarrow \theta (1 - \eta \lambda) \\
    m &\leftarrow \beta_1 m + (1 - \beta_1) g \\
    v &\leftarrow \beta_2 v + (1 - \beta_2) g^2 \\
    \hat m &= m / (1 - \beta_1^t), \quad \hat v = v / (1 - \beta_2^t) \\
    \theta &\leftarrow \theta - \eta\, \hat m / (\sqrt{\hat v} + \epsilon)
    \end{aligned}
    $$

    The state layout (``step`` as a CPU float tensor, ``exp_avg``,
    ``exp_avg_sq``, optional ``max_exp_avg_sq``) matches ``torch.optim.AdamW``,
    so checkpoints saved by one implementation load into the other. State is
    created lazily on the first step, so the optimizer can be built before the
    model is moved to the GPU (as the training scripts do).

    Not supported (on purpose, to keep the update loop plain): ``foreach``,
    ``fused``, ``capturable``, ``differentiable`` and sparse gradients.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
                 amsgrad=False, maximize=False):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if eps < 0.0:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, maximize=maximize)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            maximize = group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                if maximize:
                    grad = -grad

                state = self.state[p]
                if len(state) == 0:
                    # same layout as torch.optim.AdamW so state dicts are interchangeable
                    state['step'] = torch.tensor(0.0, dtype=torch.float32)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step'].item()

                # decoupled weight decay: shrink the weights directly, independent of the
                # adaptive step (this is what distinguishes AdamW from Adam + L2)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # first and second moment estimates
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def use_torch_adamw():
    """True if the USE_TORCH_ADAMW env var is set to anything but an off-value."""
    return os.environ.get('USE_TORCH_ADAMW', '').strip().lower() not in ('', '0', 'false', 'no', 'off')


def build_adamw(params, **kwargs):
    """AdamW factory: the custom implementation above by default, or
    ``torch.optim.AdamW`` if the USE_TORCH_ADAMW env var is set (e.g. ``USE_TORCH_ADAMW=1``).

    Both take the same constructor arguments. Which one was picked is logged on
    the 'global' logger, which the training scripts configure.
    """
    cls = torch.optim.AdamW if use_torch_adamw() else AdamW
    optimizer = cls(params, **kwargs)
    logging.getLogger('global').info(
        'Optimizer: {}.{} ({})'.format(cls.__module__, cls.__name__,
                                       'USE_TORCH_ADAMW set' if cls is torch.optim.AdamW else 'custom, set USE_TORCH_ADAMW=1 for torch.optim.AdamW'))
    return optimizer
