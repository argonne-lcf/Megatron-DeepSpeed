import os
import torch
import math
import torch.distributed as dist
from torch import Tensor


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
# and https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        # Initialize the base optimizer with all parameter groups
        super(Muon, self).__init__(params, defaults)

        # Process parameter groups to determine which will use Muon and which will use AdamW
        for i, group in enumerate(self.param_groups):
            # Ensure each group has all required parameters with defaults
            for k, v in defaults.items():
                group.setdefault(k, v)

            # Mark parameters as using Muon or AdamW
            group["use_muon_list"] = []

            for p_idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                use_muon = p.ndim == 2

                # If parameter is 2D but has a large dimension, it's likely an embedding or LM head, need to change this!!!
                if use_muon and max(p.shape) > 10000:
                    use_muon = False

                group["use_muon_list"].append(use_muon)

                # Initialize parameter state
                state = self.state[p]
                if len(state) == 0:
                    state["use_muon"] = use_muon
                    if use_muon:
                        state["momentum_buffer"] = torch.zeros_like(p.grad)
                    else:
                        state["step"] = torch.tensor(0.0)
                        state["moment1"] = torch.zeros_like(p.grad)
                        state["moment2"] = torch.zeros_like(p.grad)

    def __setstate__(self, state):
        """
        Handle state loading for the optimizer.
        """
        super(Muon, self).__setstate__(state)

        # Ensure all parameter groups have the required defaults
        for group in self.param_groups:
            group.setdefault("nesterov", True)
            group.setdefault("momentum", 0.95)
            group.setdefault("ns_steps", 5)
            group.setdefault("adamw_betas", (0.95, 0.95))
            group.setdefault("adamw_eps", 1e-8)
            group.setdefault("wd", 0.1)
            group.setdefault("lr", 1e-3)
            group.setdefault("use_muon_list", [])

        # Convert step from float to tensor if needed
        state_values = list(self.state.values())
        if state_values and "step" in state_values[0]:
            step_is_tensor = torch.is_tensor(state_values[0]["step"])
            if not step_is_tensor:
                for s in state_values:
                    if "step" in s:
                        s["step"] = torch.tensor(float(s["step"]))

    def adjust_lr_for_muon(self, lr, param_shape):
        """
        Adjust learning rate based on parameter shape for Muon.
        """
        A, B = param_shape[:2]
        # We adjust the learning rate based on the size of the parameter matrix
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            adamw_betas = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]

            # Get use_muon_list (initialize if not present)
            if "use_muon_list" not in group or not group["use_muon_list"]:
                group["use_muon_list"] = []
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    # Default determination of whether to use Muon
                    use_muon = p.ndim == 2
                    # Don't use Muon for embeddings/LM heads (approximated by large dimension check)
                    if use_muon and max(p.shape) > 10000:
                        use_muon = False
                    group["use_muon_list"].append(use_muon)

            # Apply optimization step to each parameter
            for param_idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad

                # Initialize state if needed
                state = self.state[p]
                if len(state) == 0:
                    # Determine if we should use Muon for this parameter
                    use_muon = False
                    if p.ndim == 2:
                        # Don't use Muon for embeddings/LM heads (approximated by large dimension check)
                        use_muon = max(p.shape) <= 10000

                    state["use_muon"] = use_muon

                    if use_muon:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    else:
                        state["step"] = torch.tensor(0.0)
                        state["moment1"] = torch.zeros_like(grad)
                        state["moment2"] = torch.zeros_like(grad)

                # Check if we should use Muon for this parameter
                use_muon = state.get("use_muon", False)

                if use_muon:
                    # Muon optimization
                    if grad.ndim > 2:
                        grad = grad.view(grad.size(0), -1)

                    # Initialize momentum buffer if not present
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)

                    if nesterov:
                        g = grad.add(buf, alpha=momentum)
                    else:
                        g = buf

                    u = zeropower_via_newtonschulz5(g, steps=ns_steps)

                    # Scale update
                    adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                    # Apply weight decay
                    p.data.mul_(1 - lr * wd)

                    # Apply update
                    p.data.add_(u, alpha=-adjusted_lr)
                else:
                    # AdamW optimization
                    beta1, beta2 = adamw_betas
                    eps = adamw_eps
                    weight_decay = wd

                    # Initialize AdamW state if not present
                    if "step" not in state:
                        state["step"] = torch.tensor(0.0)
                        state["moment1"] = torch.zeros_like(grad)
                        state["moment2"] = torch.zeros_like(grad)

                    state["step"] += 1
                    step = state["step"]
                    buf1 = state["moment1"]
                    buf2 = state["moment2"]

                    buf1.lerp_(grad, 1 - beta1)
                    buf2.lerp_(grad.square(), 1 - beta2)

                    g = buf1 / (eps + buf2.sqrt())

                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    scale = bias_correction1 / bias_correction2**0.5

                    p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(g, alpha=-lr / scale)

        return loss
