import os
import torch
import math
import torch.distributed as dist
from torch import Tensor
from typing import Iterable, Optional, Callable, List, Dict, Any

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
        adjuster_lr_ref=True #original moonlight lr adjustment
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

                use_muon = (p.ndim == 2)

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
        #adjusted_ratio = max(1.0, float(A) / float(B)) ** 0.5 #Muon implementation that enables MuP, Uncomment to have MuP enabled for Muon
        adjusted_ratio = 0.2 * math.sqrt(max(A, B)) #comment if enabling above line is uncommented
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def adjust_lr_for_muonclip(self, lr, param_shape):
        """
        Adjust learning rate based on parameter shape for Muon.
        """
        A, B = param_shape[:2]
        # We adjust the learning rate based on the size of the parameter matrix
        adjusted_ratio = max(1.0, float(A) / float(B)) ** 0.5
        #adjusted_ratio = 0.2 * math.sqrt(max(A, B))
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
                    if adjuster_lr_ref==True:
                        adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                    else:
                        adjusted_lr = self.adjust_lr_for_muonclip(lr, p.shape)

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


# ==========================================================
#            MuonClip: Muon + post-step QK-clip
# ==========================================================

class QKInputRecorder:
    """
    Minimal helper to capture the input 'x' that feeds an attention block's
    q/k projections on each forward pass.

    Usage:
      rec = QKInputRecorder()
      optimizer.attach_to_attention(attn, recorder=rec, q_attr='q_proj', k_attr='k_proj',
                                    d_head=head_dim, t=100.0, alpha=0.5)
    """
    def __init__(self, auto_clear: bool = True):
        self._buffers: Dict[int, Tensor] = {}
        self._handles: List[Any] = []
        self._auto_clear = auto_clear
        self._retrieved: set = set()  # Track which buffers were retrieved

    def _make_hook(self, key: int):
        def _capture(module, inputs):
            # forward_pre_hook → inputs[0] is the module input (x)
            x: Tensor = inputs[0]
            self._buffers[key] = x.detach()
            # Mark as not yet retrieved
            if key in self._retrieved:
                self._retrieved.remove(key)
        return _capture

    def attach(self, module) -> Callable[[], Optional[Tensor]]:
        key = id(module)
        handle = module.register_forward_pre_hook(self._make_hook(key))
        self._handles.append(handle)
        
        def getter() -> Optional[Tensor]:
            tensor = self._buffers.get(key, None)
            if tensor is not None:
                self._retrieved.add(key)
            return tensor
        return getter
    
    def clear_buffers(self):
        """Clear only retrieved buffers to free memory."""
        if self._auto_clear:
            for key in self._retrieved:
                if key in self._buffers:
                    del self._buffers[key]
            self._retrieved.clear()

    def remove(self):
        """Remove all hooks and clear buffers."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._buffers.clear()
        self._retrieved.clear()

class MuonClip(Muon):
    """
    Muon optimizer with qk-clip:
      After the normal Muon/AdamW updates, for each registered (W_q, W_k) pair:
        η = min(t / max_ij(q_i^T k_j), 1),  q = x W_q^T,  k = x W_k^T
        W_q ← η^α W_q,  W_k ← η^(1-α) W_k
      (optionally divide logits by sqrt(d_head) to match attention)

    Register pairs via:
      - attach_to_attention(attn_module, recorder, q_attr='q_proj', k_attr='k_proj', d_head=..., t=..., alpha=...)
      - register_qk_pair(W_q, W_k, x_getter, d_head=None, t=None, alpha=None)

    Notes:
      * Clipping runs under no_grad and does not backprop.
      * If x_getter() returns None (no forward this step), clipping is skipped.
      * In DDP, we take the MAX of max-logit across ranks before computing η.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1e-3,
        wd: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas=(0.95, 0.95),
        adamw_eps: float = 1e-8,
        adjuster_lr_ref=False,
        # MuonClip extras:
        qk_clip: bool = True,
        clip_t: float = 100.0,
        alpha: float = 0.5,
        use_sqrt_d: bool = True,
    ):
        super().__init__(
            params,
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adjuster_lr_ref=adjuster_lr_ref,
        )
        self._clip_enabled = bool(qk_clip)
        self._clip_t_default = float(clip_t)
        self._alpha_default = float(alpha)
        self._use_sqrt_d = bool(use_sqrt_d)
        self._pairs: List[Dict[str, Any]] = []  # W_q, W_k, x_getter, d_head, t, alpha

    # --------- Registration APIs ---------

    def register_qk_pair(
        self,
        W_q: Tensor,
        W_k: Tensor,
        x_getter: Callable[[], Optional[Tensor]],
        *,
        d_head: Optional[int] = None,
        t: Optional[float] = None,
        alpha: Optional[float] = None,
    ):

        """Register a Q/K weight pair for clipping."""
        # Validate tensor shapes
        assert W_q.ndim == 2 and W_k.ndim == 2, "W_q and W_k must be 2D tensors"
        assert W_q.size(1) == W_k.size(1), "W_q and W_k must have same input dimension"

        self._pairs.append(
            dict(
                W_q=W_q,
                W_k=W_k,
                x_getter=x_getter,
                d_head=d_head,
                t=float(t) if t is not None else None,
                alpha=float(alpha) if alpha is not None else None,
            )
        )

    def attach_to_attention(
        self,
        attn_module: torch.nn.Module,
        *,
        recorder: Optional[QKInputRecorder] = None,
        q_attr: str = "q_proj",
        k_attr: str = "k_proj",
        d_head: Optional[int] = None,
        t: Optional[float] = None,
        alpha: Optional[float] = None,
    ):
        """
        Convenience hook for the common case with separate q/k Linear modules.
        """
        assert hasattr(attn_module, q_attr) and hasattr(attn_module, k_attr), \
            f"Module must have {q_attr} and {k_attr}"
        q_lin = getattr(attn_module, q_attr)
        k_lin = getattr(attn_module, k_attr)
        assert hasattr(q_lin, "weight") and hasattr(k_lin, "weight")

        if recorder is None:
            recorder = QKInputRecorder()
        x_getter = recorder.attach(attn_module)
        self.register_qk_pair(q_lin.weight, k_lin.weight, x_getter,
                              d_head=d_head, t=t, alpha=alpha)
        return recorder  # keep this object alive somewhere!

    # ------------- Core step --------------

    @torch.no_grad()
    def _apply_qk_clip_once(
        self,
        W_q: Tensor,
        W_k: Tensor,
        x: Tensor,
        *,
        d_head: Optional[int],
        t: float,
        alpha: float,
        eps: float = 1e-12,
    ):
        """
        Compute η from current batch x and rescale W_q/W_k in-place.
        """
        # Validate and prepare input
        if x.ndim == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        assert x.ndim == 3, f"Expected 3D input (batch, seq, dim), got shape {x.shape}"
        assert x.size(-1) == W_q.size(1), \
            f"Input dim {x.size(-1)} doesn't match weight dim {W_q.size(1)}"

        # x: (B, T, d_model); W_q/W_k: (out_features, d_model)
        device = W_q.device
        x = x.to(device, non_blocking=True)

        q = x @ W_q.T
        k = x @ W_k.T

        scores = torch.einsum("bid,bjd->bij", q, k)
        if self._use_sqrt_d:
            scores = scores / (W_q.size(0) ** 0.5)
            #denom = float(d_head if d_head is not None else W_q.size(0))
            #scores = scores / (denom ** 0.5)

        # Find maximum score
        max_score = scores.max()
        
        # Check for numerical issues
        if not torch.isfinite(max_score):
            return False, float('nan')

        # Global max across DDP ranks if in distributed training
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(max_score, op=dist.ReduceOp.MAX)

        max_score_val = max_score.item() if torch.is_tensor(max_score) else max_score
        
        # Apply clipping if max score exceeds threshold
        if max_score_val > t:
            eta = t / (max_score_val + eps)
            scale_q = eta ** alpha
            scale_k = eta ** (1 - alpha)
            
            # Scale the weights
            W_q.mul_(scale_q)
            W_k.mul_(scale_k)
            
            return True, max_score_val
        
        return False, max_score_val

    @torch.no_grad()
    def step(self, closure=None):
        # 1) normal Muon/AdamW step
        loss = super().step(closure=closure)

        # 2) qk-clip (post-step)
        if self._clip_enabled and len(self._pairs) > 0:
            for pair in self._pairs:
                x = pair["x_getter"]()
                if x is None:
                    continue  # no forward for this module this step
                try:
                    clipped, max_score = self._apply_qk_clip_once(
                        pair["W_q"],
                        pair["W_k"],
                        x,
                        d_head=pair.get("d_head", None),
                        t=pair.get("t", self._clip_t_default) or self._clip_t_default,
                        alpha=pair.get("alpha", self._alpha_default)
                              if pair.get("alpha", None) is not None else self._alpha_default,
                    )
                    
                    # Optional: log clipping events for monitoring
                    # if clipped:
                    #     print(f"QK-clip triggered! Max score: {max_score:.2f} > {pair.get('t', self._clip_t_default):.2f}")
                    
                except Exception as e:
                    # Log but don't crash training
                    print(f"Warning: QK-clip failed with error: {e}")
                    continue
            
            # Clear recorder buffers after each step to prevent memory accumulation
            for recorder in self._recorders:
                recorder.clear_buffers()
        return loss
    def __del__(self):
        """Cleanup hooks when optimizer is destroyed."""
        for recorder in self._recorders:
            try:
                recorder.remove()
            except:
                pass  # Ignore errors during cleanup
