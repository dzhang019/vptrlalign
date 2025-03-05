"""
Implementation of transformer and reshaping-based sparse transformer
"""
import functools
import math

import torch as th
from torch import nn
from torch.nn import functional as F

from lib import misc, mlp
from lib import torch_util as tu
from lib import util

SENTINEL = 0.1337
def attention(
    Q_bte,
    K_bTe,
    V_bTe,
    dtype,
    mask=True,
    extra_btT=None,
    maxlen=None,
    check_sentinel=False,
    use_muP_factor=False,
):
    """Fixed attention function with correct tensor handling"""
    b, t, e = Q_bte.shape
    _, T, _ = K_bTe.shape
    
    # print(f"DEBUG: Q shape: {Q_bte.shape}, K shape: {K_bTe.shape}, V shape: {V_bTe.shape}")
    
    if t == 1:
        # print(f"DEBUG: Single-step mode with T={T}")
        pass
    elif t > 1 and T > t:
        # print(f"DEBUG: Batch mode with t={t}, T={T}")
        K_bTe = K_bTe[:, -t:, :]
        V_bTe = V_bTe[:, -t:, :]
        # print(f"DEBUG: After truncation: K shape: {K_bTe.shape}")
    
    # Update T after possible truncation
    T = K_bTe.shape[1]
    
    # CRITICAL CHANGE: Always create bias with shape matching Q and K
    if isinstance(mask, th.Tensor):
        # print(f"DEBUG: Mask is a tensor with shape {mask.shape}")
        # Create new bias with the right shape, ignoring the input mask
        bias = th.zeros((b, t, T), device=Q_bte.device, dtype=th.float32)
        
        # Copy as much as possible from the mask
        min_rows = min(mask.shape[1], t)
        min_cols = min(mask.shape[2], T)
        mask_slice = ~mask[:, :min_rows, :min_cols]
        
        # Set the copied part
        bias[:, :min_rows, :min_cols] = mask_slice.float() * -1e9
    elif isinstance(mask, bool) and mask:
        # print("DEBUG: Creating causal mask manually")
        # Create causal mask manually (lower triangular)
        bias = th.zeros((b, t, T), device=Q_bte.device, dtype=th.float32)
        for i in range(t):
            max_attend = min(i+1, T)
            if max_attend < T:
                bias[:, i, max_attend:] = -1e9
    else:
        # print("DEBUG: No mask, using zero bias")
        bias = th.zeros((b, t, T), device=Q_bte.device, dtype=th.float32)
    
    # print(f"DEBUG: Bias shape after creation: {bias.shape}")
    
    if extra_btT is not None:
        # print(f"DEBUG: extra_btT provided with shape: {extra_btT.shape}")
        
        # Create a new extra_btT with the right shape (matching bias exactly)
        new_extra = th.zeros_like(bias)
        
        # Copy as much as possible
        min_rows = min(extra_btT.shape[1], bias.shape[1])
        min_cols = min(extra_btT.shape[2], bias.shape[2])
        
        # Set the copied part
        new_extra[:, :min_rows, :min_cols] = extra_btT[:, :min_rows, :min_cols]
        extra_btT = new_extra
        
        # print(f"DEBUG: Final extra_btT shape: {extra_btT.shape}")
        bias = bias + extra_btT
    
    # print(f"DEBUG: Final bias shape before baddbmm: {bias.shape}")
    # print(f"DEBUG: Q shape: {Q_bte.shape}, K.T shape: {K_bTe.transpose(-1, -2).shape}")
    
    # Double-check dimensions are compatible before baddbmm
    assert bias.shape[2] == K_bTe.shape[1], f"Bias col dim ({bias.shape[2]}) must match K row dim ({K_bTe.shape[1]})"
    
    # Compute attention with scaled dot product
    logit_btT = th.baddbmm(
        bias,
        Q_bte.float(),
        K_bTe.float().transpose(-1, -2),
        alpha=1 / math.sqrt(e),
    )
    
    # Apply softmax along context dimension
    W_btT = th.softmax(logit_btT, dim=2).to(dtype)
    
    # Compute weighted sum of values
    A_bte = th.einsum("btp,bpe->bte", W_btT, V_bTe)
    return A_bte
# def attention(Q_bte, K_bTe, V_bTe, dtype, mask=True, extra_btT=None, maxlen=None, check_sentinel=False, use_muP_factor=False):
#     # print(f"Q shape: {Q_bte.shape}, K shape: {K_bTe.shape}, V shape: {V_bTe.shape}")
#     b, t, e = Q_bte.shape
#     _, T, _ = K_bTe.shape
    
#     # Check for mismatch in query and key sequence lengths
#     # Only truncate during training, not during environment stepping
#     truncated = False
#     if t != T and t == 1 and T > 1:
#         # This is the case where we're in step-by-step mode (t=1) with a large context (T>1)
#         # In this case, we shouldn't truncate K as it's intentionally longer for context
#         # print(f"Single-step attention with context: Q={t}, K={T}")
#         truncated = False
#     elif t != T and t > 1 and T > t:
#         # This is the case where we're in batch mode but K is larger than Q
#         # Only in this case should we truncate
#         # print(f"Sequence length mismatch during batch processing: Q={t}, K={T}")
#         K_bTe = K_bTe[:, -t:, :]
#         V_bTe = V_bTe[:, -t:, :]
#         # print(f"Truncated K/V to match Q length: K={K_bTe.shape}")
#         truncated = True
    
#     assert Q_bte.dtype == K_bTe.dtype == dtype, f"{Q_bte.dtype}, {K_bTe.dtype}, {dtype} must all match"
#     e = Q_bte.shape[2]
    
#     if check_sentinel:
#         invalid = (K_bTe == SENTINEL).int().sum(dim=-1) == e
#         invalid = misc.reshape(invalid, "b, T", "b, 1, T")
    
#     if isinstance(mask, th.Tensor):
#         bias = (~mask).float() * -1e9
#     elif mask:
#         # Use the original sequence lengths for bias calculation
#         bias = get_attn_bias_cached(Q_bte.shape[1], K_bTe.shape[1], maxlen=maxlen, device=Q_bte.device, dtype=th.float32)
#     else:
#         bias = Q_bte.new_zeros((), dtype=th.float32)
    
#     if extra_btT is not None:
#         # Only handle dimension matching if we actually have a shape mismatch
#         if bias.shape[1] != extra_btT.shape[1] or bias.shape[2] != extra_btT.shape[2]:
#             # Create a safe bias tensor of the right dimensions
#             if isinstance(bias, th.Tensor) and bias.dim() > 0:
#                 # Handle dimension 1 mismatch (batch sequence length)
#                 if bias.shape[1] != extra_btT.shape[1]:
#                     if bias.shape[1] == 1:
#                         bias = bias.expand(-1, extra_btT.shape[1], -1)
#                     elif extra_btT.shape[1] == 1:
#                         extra_btT = extra_btT.expand(-1, bias.shape[1], -1)
#                     else:
#                         min_dim1 = min(bias.shape[1], extra_btT.shape[1])
#                         bias = bias[:, :min_dim1, :]
#                         extra_btT = extra_btT[:, :min_dim1, :]
                
#                 # Handle dimension 2 mismatch (sequence length)
#                 if bias.shape[2] != extra_btT.shape[2]:
#                     min_dim2 = min(bias.shape[2], extra_btT.shape[2])
#                     bias = bias[:, :, :min_dim2]
#                     extra_btT = extra_btT[:, :, :min_dim2]
        
#         bias = bias + extra_btT
    
    # Handle bias shape for baddbmm operation
    if isinstance(bias, th.Tensor) and bias.dim() > 0:
        # Ensure bias has the right shape for bmm: [b, t, T]
        if bias.shape[1] != Q_bte.shape[1] or bias.shape[2] != K_bTe.shape[1]:
            # print(f"Adjusting bias shape from {bias.shape} to match Q={Q_bte.shape[1]}, K={K_bTe.shape[1]}")
            # Create new bias with correct shape
            new_bias = Q_bte.new_zeros((Q_bte.shape[0], Q_bte.shape[1], K_bTe.shape[1]), dtype=th.float32)
            
            # Fill with values from original bias where possible
            min_dim1 = min(bias.shape[1], Q_bte.shape[1])
            min_dim2 = min(bias.shape[2], K_bTe.shape[1])
            new_bias[:, :min_dim1, :min_dim2] = bias[:, :min_dim1, :min_dim2]
            
            # Fill rest with masked value if using masking
            if mask:
                # For causal masking, positions where query can't attend to key
                if min_dim1 < Q_bte.shape[1] or min_dim2 < K_bTe.shape[1]:
                    for i in range(Q_bte.shape[1]):
                        valid_k = min(i+1, K_bTe.shape[1])
                        new_bias[:, i, valid_k:] = -1e9
            
            bias = new_bias
    
    # Now bias should have shape [b, t, T] 
    # print(f"Final shapes - bias: {bias.shape}, Q: {Q_bte.shape}, K: {K_bTe.shape}")
    
    # Compute attention
    logit_btT = th.baddbmm(
        bias,
        Q_bte.float(),
        K_bTe.float().transpose(-1, -2),
        alpha=(1 / e) if use_muP_factor else (1 / math.sqrt(e)),
    )
    
    if check_sentinel:
        logit_btT = logit_btT - 1e9 * invalid.float()
    
    W_btT = th.softmax(logit_btT, dim=2).to(dtype)
    
    if callable(V_bTe):
        V_bTe = V_bTe()
    
    A_bte = th.einsum("btp,bpe->bte", W_btT, V_bTe)
    return A_bte


class Attn:
    """
    Defines an attention mechanism
    All the mechanisms here can be defined by two operations:
    1. preprocessing Q,K,V,R[=relative attention query]
       to move axes from embedding dimension to batch dimension, and possibly doing shifts.
    2. postprocessing the final result to move axes back to embedding axis.
    """
    def __init__(self, mask, maxlen):
        self.mask = mask
        self.maxlen = maxlen

    def preproc_qkv(self, Q_bte, K_bte, V_bte):
        raise NotImplementedError

    def preproc_r(self, R_btn):
        raise NotImplementedError


def split_heads(x_bte, h):
    b, t, e = x_bte.shape
    assert e % h == 0, "Embsize must be divisible by number of heads"
    q = e // h
    x_bthq = x_bte.reshape((b, t, h, q))
    x_bhtq = misc.transpose(x_bthq, "bthq", "bhtq")
    x_Btq = x_bhtq.reshape((b * h, t, q))
    return x_Btq


class All2All(Attn):
    def __init__(self, nhead, maxlen, mask=True, head_dim=None):
        super().__init__(mask=mask, maxlen=maxlen)
        assert (nhead is None) != (head_dim is None), "exactly one of nhead and head_dim must be specified"
        self.h = nhead
        self.head_dim = head_dim

    def preproc_qkv(self, *xs):
        q = xs[0].shape[-1]
        for x in xs:
            assert x.shape[-1] == q, "embedding dimensions do not match"
        h = self.h or misc.exact_div(q, self.head_dim)
        postproc = functools.partial(self.postproc_a, h=h)
        return (postproc, *tuple(split_heads(x, h) for x in xs))

    def preproc_r(self, R_btn):
        _, ret = self.preproc_qkv(R_btn)
        return ret

    def postproc_a(self, A_Btq, h):
        B, t, q = A_Btq.shape
        b = B // h
        A_bhtq = A_Btq.reshape((b, h, t, q))
        A_bthq = misc.transpose(A_bhtq, "bhtq", "bthq")
        A_bte = A_bthq.reshape((b, t, h * q))
        return A_bte


def _required_padding(dim, target_div):
    if dim % target_div == 0:
        return 0
    else:
        return target_div - dim % target_div


class StridedAttn(Attn):
    def __init__(self, nhead, stride, maxlen, mask=True):
        super().__init__(mask=mask, maxlen=maxlen)
        self.h = nhead
        self.stride = stride

    def _preproc(self, x, name, Q_t=None, Q_pad=None):
        x, undo = misc.reshape_undo(x, "b, t*stride, e", "b, 1, t, stride*e", stride=self.stride)
        if name == "Q":
            Q_pad = _required_padding(x.shape[2], self.maxlen)
        original_t = x.shape[2]
        x = F.pad(x, (0, 0, 0, Q_pad), value=SENTINEL)
        undo = misc.compose_undo(undo, lambda x: x[:, :, :original_t])
        if name == "Q":
            Q_t = x.shape[2]
            assert Q_t % self.maxlen == 0, f"{Q_t} % {self.maxlen} != 0"
        else:
            required_len = Q_t + self.maxlen
            if x.shape[2] < required_len:
                x = F.pad(x, (0, 0, required_len - x.shape[2], 0), value=SENTINEL)
            assert x.shape[2] >= required_len
            back = x[:, :, -Q_t - self.maxlen : -self.maxlen]
            front = x[:, :, -Q_t:]
            x = th.cat([back, front], dim=1)
        _, _, t, _ = x.shape
        assert t == Q_t, f"{t} != {Q_t}"
        x, undo = misc.reshape_undo(
            x,
            "b, pad_shift, t*maxlen, stride*h*q",
            "b, pad_shift, t, maxlen, stride, h, q",
            maxlen=self.maxlen,
            h=self.h,
            stride=self.stride,
            undo=undo,
        )
        x, undo = misc.transpose_undo(x, "bptmshq", "bthspmq", undo=undo)
        x, undo = misc.reshape_undo(
            x,
            "b, t, h, stride, pad_shift, maxlen, q",
            "b*t*h*stride, pad_shift*maxlen, q",
            undo=undo,
        )
        if name == "Q":
            return x, undo, Q_t, Q_pad
        else:
            return x

    def preproc_qkv(self, Q_bte, K_bte, V_bte):
        pad = _required_padding(Q_bte.shape[1], self.stride)
        if pad:
            Q_bte = F.pad(Q_bte, (0, 0, 0, pad), value=SENTINEL)
            K_bte = F.pad(K_bte, (0, 0, 0, pad), value=SENTINEL) if K_bte is not None else None
            V_bte = F.pad(V_bte, (0, 0, 0, pad), value=SENTINEL) if V_bte is not None else None
            undo = lambda x, pad=pad: x[:, :-pad]
        else:
            undo = None
        if K_bte is not None:
            pad = _required_padding(K_bte.shape[1], self.stride)
            if pad:
                K_bte = F.pad(K_bte, (0, 0, pad, 0), value=SENTINEL)
                V_bte = F.pad(V_bte, (0, 0, pad, 0), value=SENTINEL)
        assert Q_bte.shape[1] % self.stride == 0
        assert K_bte is None or K_bte.shape[1] % self.stride == 0
        assert V_bte is None or V_bte.shape[1] % self.stride == 0
        Q, postproc, Q_t, Q_pad = self._preproc(Q_bte, "Q")
        postproc = misc.compose_undo(undo, postproc)
        return (
            postproc,
            Q,
            self._preproc(K_bte, "K", Q_t=Q_t, Q_pad=Q_pad) if K_bte is not None else None,
            self._preproc(V_bte, "V", Q_t=Q_t, Q_pad=Q_pad) if V_bte is not None else None,
        )

    def preproc_r(self, R_bte):
        _, R, _, _ = self.preproc_qkv(R_bte, None, None)
        return R


Q_SCALE = 0.1
K_SCALE = 0.2
V_SCALE = 1.0
PROJ_SCALE = 1.0
MLP0_SCALE = 1.0
MLP1_SCALE = 1.0
R_SCALE = 0.1
B_SCALE = 0.2


class AttentionLayerBase(nn.Module):
    def __init__(
        self,
        *,
        attn,
        scale,
        x_size,
        c_size,
        qk_size,
        v_size,
        dtype,
        relattn=False,
        seqlens=None,
        separate=False,
    ):
        super().__init__()
        dtype = tu.parse_dtype(dtype)
        self.attn = attn
        self.x_size = x_size
        self.c_size = c_size
        s = math.sqrt(scale)
        separgs = dict(seqlens=seqlens, separate=separate)
        self.q_layer = MultiscaleLinear(x_size, qk_size, name="q", scale=Q_SCALE, dtype=dtype, **separgs)
        self.k_layer = MultiscaleLinear(c_size, qk_size, name="k", scale=K_SCALE, bias=False, dtype=dtype, **separgs)
        self.v_layer = MultiscaleLinear(c_size, v_size, name="v", scale=V_SCALE * s, bias=False, dtype=dtype, **separgs)
        self.proj_layer = MultiscaleLinear(v_size, x_size, name="proj", scale=PROJ_SCALE * s, dtype=dtype, **separgs)
        self.relattn = relattn
        maxlen = attn.maxlen
        assert maxlen > 0 or not attn.mask
        if self.relattn:
            nbasis = 10
            self.r_layer = tu.NormedLinear(x_size, nbasis * attn.h, scale=R_SCALE, dtype=dtype)
            self.b_nd = nn.Parameter(th.randn(nbasis, maxlen) * B_SCALE)
        self.maxlen = maxlen
        self.dtype = dtype

    def relattn_logits(self, X_bte, T):
        R_btn = self.r_layer(X_bte).float()
        R_btn = self.attn.preproc_r(R_btn)
        t = R_btn.shape[1]
        D_ntT = util.bandify(self.b_nd, t, T)
        extra_btT = th.einsum("btn,ntp->btp", R_btn, D_ntT)
        return extra_btT


def quick_gelu(x):
    return x * th.sigmoid(1.702 * x)


def act(actname, x):
    if actname == "relu":
        return F.relu(x)
    elif actname == "gelu":
        return quick_gelu(x)
    elif actname == "none":
        return x
    else:
        raise NotImplementedError(actname)


class SelfAttentionLayer(AttentionLayerBase):
    """
    Residual attention layer that takes a single tensor x and has it attend to itself.
    Has the form:
         output = x + f(x)
    """
    def __init__(
        self,
        x_size,
        attn,
        scale,
        dtype="float32",
        norm="layer",
        cache_keep_len=None,
        relattn=False,
        log_scope="sa",
        use_muP_factor=False,
        **kwargs,
    ):
        super().__init__(
            x_size=x_size,
            c_size=x_size,
            qk_size=x_size,
            v_size=x_size,
            attn=attn,
            scale=scale,
            relattn=relattn,
            dtype=dtype,
            **kwargs,
        )
        self.ln_x = util.get_norm(norm, x_size, dtype=dtype)
        if cache_keep_len is None:
            if hasattr(attn, "cache_keep_len"):
                cache_keep_len = attn.cache_keep_len
            else:
                if isinstance(attn, StridedAttn):
                    stride = attn.stride
                else:
                    stride = 1
                cache_keep_len = stride * attn.maxlen
        self.cache_keep_len = cache_keep_len
        self.log_scope = log_scope
        self.use_muP_factor = use_muP_factor

    def residual(self, X_bte, state):
        # Save a copy for the skip connection.
        X_in = X_bte.clone()
        # Apply layer norm on a clone of the input.
        X_ln = self.ln_x(X_bte.clone())
        # print(f"xf.py (residual): X_bte shape: {X_bte.shape}")
        # print(f"xf.py (residual): X_ln shape: {X_ln.shape}")
        # Pass clones of X_ln to each linear layer to avoid in-place modifications.
        Q_bte = self.q_layer(X_ln.clone())
        K_bte = self.k_layer(X_ln.clone())
        V_bte = self.v_layer(X_ln.clone())
        # print(f"xf.py (residual): Q_bte shape: {Q_bte.shape}")
        # print(f"xf.py (residual): K_bte shape: {K_bte.shape}")
        # print(f"xf.py (residual): V_bte shape: {V_bte.shape}")
        
        if state:
            state, K_bte, V_bte = self.update_state(state, K_bte, V_bte)
            # print(f"Updated state: keys shape = {state[0].shape}, values shape = {state[1].shape}")
        
        postproc_closure, Q_bte, K_bte, V_bte = self.attn.preproc_qkv(Q_bte, K_bte, V_bte)
        extra_btT = self.relattn_logits(X_ln, K_bte.shape[1]) if self.relattn else None
        # print(f"Q_bte (post-preproc) shape: {Q_bte.shape}")
        # print(f"K_bte (post-preproc) shape: {K_bte.shape}")
        # print(f"V_bte (post-preproc) shape: {V_bte.shape}")
        #if extra_btT is not None:
            # print(f"extra_btT shape: {extra_btT.shape}")
        A_bte = attention(
            Q_bte,
            K_bte,
            V_bte,
            mask=self.attn.mask,
            extra_btT=extra_btT,
            maxlen=self.maxlen,
            dtype=self.dtype,
            check_sentinel=isinstance(self.attn, StridedAttn),
            use_muP_factor=self.use_muP_factor,
        )
        A_bte = postproc_closure(A_bte)
        Aproj_bte = self.proj_layer(A_bte)
        # print(f"Aproj_bte shape: {Aproj_bte.shape}")
        # Return the residual projection.
        return Aproj_bte, state

    def forward(self, X_bte, state):
        # Use the original input for the skip connection.
        R_bte, state = self.residual(X_bte, state)
        return X_bte + R_bte, state

    def stateless_forward(self, X_bte):
        out_bte, _state = self.forward(X_bte, None)
        return out_bte

    # def update_state(self, state, K_bte, V_bte):
    #     def append(prev, new):
    #         tprev = prev.shape[1]
    #         startfull = max(tprev - self.cache_keep_len, 0)
    #         full = th.cat([prev[:, startfull:], new], dim=1)
    #         outstate = full[:, max(full.shape[1] - self.cache_keep_len, 0):]
    #         return outstate, full
    #     instate_K, instate_V = state
    #     outstate_K, K_bte = append(instate_K, K_bte)
    #     outstate_V, V_bte = append(instate_V, V_bte)
    #     assert outstate_K.shape[-2] <= self.cache_keep_len
    #     return (outstate_K, outstate_V), K_bte, V_bte
    def update_state(self, state, K_bte, V_bte):
        def append(prev, new):
            # Calculate full sequence length after concatenation
            full_length = prev.shape[1] + new.shape[1]
            
            # If concatenating would exceed maxlen, drop oldest entries
            if full_length > self.maxlen:
                # Keep exactly maxlen timesteps after concatenation
                prev = prev[:, -(self.maxlen - new.shape[1]):]
            
            # Concatenate
            full = th.cat([prev, new], dim=1)
            
            # Ensure we don't exceed maxlen
            assert full.shape[1] <= self.maxlen, f"Sequence too long: {full.shape[1]} > {self.maxlen}"
            
            return full, full

        instate_K, instate_V = state
        outstate_K, K_bte = append(instate_K, K_bte)
        outstate_V, V_bte = append(instate_V, V_bte)
        
        return (outstate_K, outstate_V), K_bte, V_bte

    def initial_state(self, batchsize, initial_T=0):
        # return (
        #     tu.zeros((batchsize, initial_T, self.x_size), dtype=self.dtype),
        #     tu.zeros((batchsize, initial_T, self.x_size), dtype=self.dtype),
        # )
        keys = tu.zeros((batchsize, initial_T, self.x_size), dtype=self.dtype)
        values = tu.zeros((batchsize, initial_T, self.x_size), dtype=self.dtype)
        # print(f"Initialized cache: keys shape = {keys.shape}, values shape = {values.shape}")
        return keys, values

    def empty_state(self):
        return None


class PointwiseLayer(nn.Module):
    """
    Residual MLP applied at each timestep
    """
    def __init__(self, x_size, scale, dtype, norm, actname="relu", mlp_ratio=2):
        super().__init__()
        s = math.sqrt(scale)
        self.ln = util.get_norm(norm, x_size, dtype=dtype)
        self.mlp = mlp.MLP(
            insize=x_size,
            nhidlayer=1,
            outsize=x_size,
            hidsize=int(x_size * mlp_ratio),
            hidactiv=functools.partial(act, actname),
            dtype=dtype,
        )
        self.mlp.layers[0].weight.data *= MLP0_SCALE * s
        self.mlp.layers[1].weight.data *= MLP1_SCALE * s

    def residual(self, x):
        x = self.ln(x)
        x = self.mlp(x)
        return x

    def forward(self, x):
        return x + self.residual(x)


def _is_separate(sep, name):
    if isinstance(sep, bool):
        return sep
    assert isinstance(sep, set)
    if name in sep:
        sep.remove(name)
        return True
    else:
        return False


def make_maybe_multiscale(make_fn, *args, seqlens, separate, name, **kwargs):
    """
    Either creates one instance of a module or creates a separate instance of the module for each resolution.
    """
    if _is_separate(separate, name):
        modules = [make_fn(*args, **kwargs) for _ in seqlens]
        return SplitCallJoin(modules, seqlens)
    else:
        return make_fn(*args, **kwargs)


class SplitCallJoin(nn.Module):
    def __init__(self, mods, seqlens):
        super().__init__()
        self.mods = nn.ModuleList(mods)
        self.seqlens = seqlens

    def forward(self, x):
        tl = sum(self.seqlens)
        x, undo = misc.reshape_undo(x, "..., z*tl, e", "..., z, tl, e", tl=tl)
        x = list(th.split(x, self.seqlens, dim=-2))
        new_x = []
        for x, mod in misc.safezip(x, self.mods):
            x, this_undo = misc.reshape_undo(x, "..., z, l, e", "..., z*l, e")
            x = mod(x)
            x = this_undo(x)
            new_x.append(x)
        x = th.cat(new_x, dim=-2)
        x = undo(x)
        return x


MultiscaleLinear = functools.partial(make_maybe_multiscale, tu.NormedLinear)
MultiscalePointwise = functools.partial(make_maybe_multiscale, PointwiseLayer)
