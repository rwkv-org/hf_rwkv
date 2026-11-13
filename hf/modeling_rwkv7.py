from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers.utils.deprecation import deprecate_kwarg

from .configuration_rwkv7 import RWKV7Config

from .rwkv7_kernels import (
    addcmul_rwkv7,
    k_update_rwkv7,
    output_correction,
    sqrelu,
    token_shift,
    wkv_chunk,
    wkv_recurrent,
)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


logger = logging.get_logger(__name__)


def _seq_lens_to_cu_seqlens(seq_lens: torch.Tensor) -> torch.Tensor:
    """Convert AutoModel THD ``seq_lens`` [B, docs] to cumulative lengths."""
    if seq_lens.dim() > 1:
        if seq_lens.shape[0] != 1:
            raise ValueError(
                "RWKV7 packed training requires batch_size=1; "
                f"got seq_lens batch={seq_lens.shape[0]}."
            )
        seq_lens = seq_lens.reshape(-1)
    lengths = seq_lens.to(dtype=torch.long)
    lengths = lengths[lengths > 0]
    if lengths.numel() == 0:
        raise ValueError("seq_lens does not contain any positive sequence length.")
    return torch.cat(
        (lengths.new_zeros(1), lengths.cumsum(0)),
        dim=0,
    )


class RWKV7CacheLayer(CacheLayerMixin):
    """Per-layer recurrent state for RWKV7.

    This follows the modern transformers cache protocol: a ``Cache`` is a
    container of ``CacheLayerMixin`` objects, and each layer owns its own
    sequence-length bookkeeping.  RWKV7 stores three tensors per layer:

    * ``recurrent_state``: the WKV matrix state
    * ``conv_state``: the previous token used by token-shift in attention
    * ``ffn_state``: the previous token used by token-shift in the FFN
    """

    is_sliding = False
    is_compileable = False

    def __init__(self) -> None:
        super().__init__()
        self.conv_state: torch.Tensor | None = None
        self.recurrent_state: torch.Tensor | None = None
        self.ffn_state: torch.Tensor | None = None
        self._seen_tokens = 0
        self.device = torch.device("cpu")

    @property
    def state(self) -> dict[str, torch.Tensor | None]:
        return {
            "conv_state": self.conv_state,
            "recurrent_state": self.recurrent_state,
            "ffn_state": self.ffn_state,
        }

    @property
    def max_batch_size(self) -> int:
        for tensor in self.state.values():
            if isinstance(tensor, torch.Tensor):
                return int(tensor.shape[0])
        return 0

    @property
    def max_cache_len(self) -> int:
        return -1

    def lazy_initialization(self, *args: Any, **kwargs: Any) -> None:
        # The states are created by the first ``update`` call.  This method
        # only exists to satisfy the CacheLayerMixin protocol (the signature
        # changed between transformers 4.x and 5.x).
        return None

    def update(
        self,
        *,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
        ffn_state: torch.Tensor | None = None,
        offset: int = 0,
        cache_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor | None]:
        del cache_kwargs
        if conv_state is not None:
            self.conv_state = conv_state
            self.device = conv_state.device
            self.is_initialized = True
        if recurrent_state is not None:
            self.recurrent_state = recurrent_state
            self.device = recurrent_state.device
            self.is_initialized = True
        if ffn_state is not None:
            self.ffn_state = ffn_state
            self.device = ffn_state.device
            self.is_initialized = True
        self._seen_tokens += int(offset)
        return self.state

    def get_seq_length(self, cache_position: torch.Tensor | None = None) -> int:
        del cache_position
        return self._seen_tokens

    def get_mask_sizes(self, query_length: int | torch.Tensor | None = None) -> tuple[int, int]:
        del query_length
        # RWKV7 is strictly causal and does not need a cache-derived mask.
        return 0, 0

    def get_max_length(self) -> int:
        return -1

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self) -> None:
        self.conv_state = None
        self.recurrent_state = None
        self.ffn_state = None
        self._seen_tokens = 0
        self.is_initialized = False

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for name in ("conv_state", "recurrent_state", "ffn_state"):
            state = getattr(self, name)
            if isinstance(state, torch.Tensor):
                setattr(self, name, state.index_select(0, beam_idx.to(state.device)))

    def batch_repeat_interleave(self, repeats: int) -> None:
        for name in ("conv_state", "recurrent_state", "ffn_state"):
            state = getattr(self, name)
            if isinstance(state, torch.Tensor):
                setattr(self, name, state.repeat_interleave(repeats, dim=0))

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for name in ("conv_state", "recurrent_state", "ffn_state"):
            state = getattr(self, name)
            if isinstance(state, torch.Tensor):
                setattr(self, name, state.index_select(0, indices))

    def offload(self) -> None:
        for name in ("conv_state", "recurrent_state", "ffn_state"):
            state = getattr(self, name)
            if isinstance(state, torch.Tensor):
                setattr(self, name, state.to("cpu", non_blocking=True))

    def prefetch(self) -> None:
        for name in ("conv_state", "recurrent_state", "ffn_state"):
            state = getattr(self, name)
            if isinstance(state, torch.Tensor) and state.device != self.device:
                setattr(self, name, state.to(self.device, non_blocking=True))


class RWKV7Cache(Cache):
    """A transformers ``Cache`` containing RWKV7 recurrent states.

    This mirrors the cache protocol used by FLA and by newer transformers
    linear-attention models: ``past_key_values`` is a ``Cache`` object, each
    layer is a ``CacheLayerMixin``, and per-layer sequence lengths are updated
    with an explicit ``offset`` (number of tokens processed in this call).

    We deliberately define the layer here instead of importing
    ``fla.models.utils.FLACache``: that cache is FLA's internal model utility,
    and its ``FLALayer`` currently has no recurrent-state ``reorder_cache``
    (beam search would fail).  Depending only on the stable
    ``transformers.cache_utils`` protocol keeps the generated HF directory
    portable across FLA versions.
    """

    def __init__(
        self,
        config: RWKV7Config | None = None,
        layers: list[RWKV7CacheLayer] | None = None,
    ) -> None:
        if layers is None and config is not None:
            layers = [
                RWKV7CacheLayer() for _ in range(config.num_hidden_layers)
            ]
        if layers is not None:
            super().__init__(layers=layers)
        else:
            super().__init__(layer_class_to_replicate=RWKV7CacheLayer)

    def update(
        self,
        layer_idx: int,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
        ffn_state: torch.Tensor | None = None,
        offset: int = 0,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | None]:
        while len(self.layers) <= layer_idx:
            self.layers.append(RWKV7CacheLayer())
        return self.layers[layer_idx].update(
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            ffn_state=ffn_state,
            offset=offset,
            cache_kwargs=cache_kwargs,
        )

    def __getitem__(self, layer_idx: int) -> dict[str, torch.Tensor | None]:
        return self.layers[layer_idx].state

    def __iter__(self):
        for layer in self.layers:
            yield layer.state

    def to_legacy_cache(self) -> tuple[dict[str, torch.Tensor | None], ...]:
        return tuple(layer.state for layer in self.layers)

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: tuple[dict[str, torch.Tensor | None], ...] | None = None,
        seen_tokens: int = 0,
    ) -> "RWKV7Cache":
        cache = cls()
        if past_key_values:
            for state in past_key_values:
                layer = RWKV7CacheLayer()
                layer.conv_state = state.get("conv_state")
                layer.recurrent_state = state.get("recurrent_state")
                layer.ffn_state = state.get("ffn_state")
                layer._seen_tokens = int(seen_tokens)
                layer.is_initialized = any(
                    value is not None for value in state.values()
                )
                cache.layers.append(layer)
        return cache


class LoRA(nn.Module):
    """Small low-rank adapter used by the RWKV7 attention block.

    This is intentionally local instead of importing ``fla.layers.rwkv6.LoRA``
    so that the generated model only depends on the kernel backend for FLA.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: bool | None = True,
        activation: str | None = "tanh",
    ) -> None:
        super().__init__()
        activations = {
            None: nn.Identity(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU(),
        }
        if activation not in activations:
            raise ValueError(f"Unsupported LoRA activation: {activation!r}")
        self.bias = bias
        self.lora = nn.Sequential(
            nn.Linear(input_dim, low_rank_dim, bias=False),
            activations[activation],
            nn.Linear(low_rank_dim, output_dim, bias=bias),
        )
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module) -> None:
        if not isinstance(module, LoRA) or getattr(module, "_is_hf_initialized", False):
            return
        if module.lora[0].weight is None or module.lora[0].weight.is_meta:
            # FSDP2 constructs the model on the meta device and loads real
            # weights afterwards; there is nothing to initialize here.
            module._is_hf_initialized = True
            return
        nn.init.zeros_(module.lora[0].weight)
        original_dtype = module.lora[2].weight.dtype
        shape = module.lora[2].weight.shape
        gain = math.sqrt(shape[1] / shape[0]) if shape[1] > shape[0] else 1.0
        weight = module.lora[2].weight.float()
        nn.init.orthogonal_(weight, gain=gain * 0.1)
        module.lora[2].weight.data.copy_(weight.to(original_dtype))
        if module.lora[2].bias is not None:
            nn.init.zeros_(module.lora[2].bias)
        module._is_hf_initialized = True

    def set_bias_value(self, value) -> None:
        if self.bias and self.lora[2].bias is not None:
            if isinstance(value, torch.Tensor):
                self.lora[2].bias.data.copy_(value.to(self.lora[2].bias.dtype))
            else:
                nn.init.constant_(self.lora[2].bias, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)


@dataclass
class RWKV7ModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    cache_params: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class RWKV7CausalLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    cache_params: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class RWKV7Attention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        head_dim: int | None = 64,
        num_heads: int | None = None,
        decay_low_rank_dim: int | None = None,
        gate_low_rank_dim: int | None = None,
        a_low_rank_dim: int | None = None,
        v_low_rank_dim: int | None = None,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        fuse_norm: bool = False,
        value_dim: int = None,
        num_hidden_layers: int = None,
        **kwargs,
    ) -> RWKV7Attention:
        super().__init__()

        self.mode = mode
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        self.hidden_size = hidden_size

        self.key_dim = hidden_size
        self.value_dim = value_dim if value_dim is not None else hidden_size
        if head_dim is None and num_heads is None:
            raise ValueError("Either `head_dim` or `num_heads` must be specified.")
        elif head_dim is not None:
            self.head_dim = head_dim
            self.num_heads = int(hidden_size // head_dim)
        elif num_heads is not None:
            self.head_dim = int(hidden_size // num_heads)
            self.num_heads = num_heads
        self.head_v_dim = int(self.value_dim // self.num_heads)

        # Increase lora dimension for headdim>64
        factor = self.head_dim / 64
        if decay_low_rank_dim is None:
            decay_low_rank_dim = max(32, int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32))
            self.decay_low_rank_dim = decay_low_rank_dim
        else:
            self.decay_low_rank_dim = decay_low_rank_dim

        if gate_low_rank_dim is None:
            gate_low_rank_dim = max(32, int(round((5 * (hidden_size**0.5)) / 32) * 32))
            self.gate_low_rank_dim = gate_low_rank_dim
        else:
            self.gate_low_rank_dim = gate_low_rank_dim

        if a_low_rank_dim is None:
            a_low_rank_dim = max(32, int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32))
            self.a_low_rank_dim = a_low_rank_dim
        else:
            self.a_low_rank_dim = a_low_rank_dim

        if v_low_rank_dim is None:
            v_low_rank_dim = max(32, int(round((1.7 * (hidden_size**0.5)) * factor / 32) * 32))
            self.v_low_rank_dim = v_low_rank_dim
        else:
            self.v_low_rank_dim = v_low_rank_dim

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.k_k = nn.Parameter(torch.zeros(self.key_dim))
        self.k_a = nn.Parameter(torch.zeros(self.key_dim))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.w_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=decay_low_rank_dim, activation='tanh')
        if self.layer_idx != 0:
            self.v_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=v_low_rank_dim, activation=None)
        self.a_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=a_low_rank_dim, activation=None)
        self.g_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=gate_low_rank_dim, activation='sigmoid', bias=False)

        # The HF artifact always uses the unfused PyTorch norm.  A fused
        # kernel backend can still override this module if needed.  The
        # ``fuse_norm`` constructor argument is kept for API compatibility.
        self.fuse_norm = False
        self.g_norm = nn.GroupNorm(
            num_groups=self.num_heads,
            num_channels=self.value_dim,
            eps=self.head_dim * norm_eps,
            affine=elementwise_affine,
        )

        try:
            from transformers.modeling_utils import _init_weights
        except ImportError:
            _init_weights = True
        if _init_weights:
            self.apply(self._initialize_weights)
        for name, module in self.named_modules():
            module._in_rwkv_module = True


    @torch.no_grad()
    @torch.compiler.disable
    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return

        # Initialize only when we're processing the RWKV7Attention module itself
        if isinstance(module, RWKV7Attention) and self.layer_idx is not None:
            if module.x_r.is_meta:
                module._is_hf_initialized = True
                return
            ratio_0_to_1 = self.layer_idx / (self.num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (self.layer_idx / self.num_hidden_layers)  # 1 to ~0

            # Create position-based initialization tensor
            ddd = torch.ones(1, 1, self.hidden_size, device=self.x_r.device)
            www = torch.zeros(self.hidden_size, device=self.x_r.device)
            zigzag = torch.zeros(self.hidden_size, device=self.x_r.device)
            linear = torch.zeros(self.hidden_size, device=self.x_r.device)
            for n in range(self.hidden_size):
                linear[n] = n / (self.hidden_size-1) - 0.5
                zigzag[n] = ((n % self.head_dim) - ((self.head_dim-1) / 2)) / ((self.head_dim-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (self.hidden_size - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)
                ddd[0, 0, n] = n / self.hidden_size

            # Initialize x_* parameters directly
            self.x_r.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_r.dtype)
            self.x_w.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_w.dtype)
            self.x_k.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)).to(self.x_k.dtype)
            self.x_v.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)).to(self.x_v.dtype)
            self.x_a.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_a.dtype)
            self.x_g.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_g.dtype)

            # Initialize k_k, k_a, r_k
            nn.init.constant_(self.k_a, 1.02)
            nn.init.constant_(self.r_k, -0.04)
            self.k_k.data.copy_((torch.zeros(self.hidden_size, device=self.k_k.device) +
                                0.71 - linear*0.1).to(self.k_k.dtype))
            # Set specific bias values for LoRA modules
            # 0.5 comes from F.softplus
            self.w_lora.set_bias_value(www + 0.5 + zigzag*2.5)
            self.a_lora.set_bias_value(-0.19 + zigzag*0.3 + linear*0.4)

            # v0 initialization - ones (for non-first layers)
            if self.layer_idx != 0:
                self.v_lora._initialize_weights(self.v_lora)
                self.v_lora.set_bias_value(0.73 - linear*0.4)

            # Initialize GroupNorm
            self.g_norm.weight.data[:] = ((self.layer_idx + 1) / self.num_hidden_layers) ** 0.7

            # Initialize Linear projections
            self._orthogonal_init(self.r_proj.weight)
            self._orthogonal_init(self.k_proj.weight, gain=0.1)
            self._orthogonal_init(self.v_proj.weight)
            self.o_proj.weight.data.zero_()

            # Clean up temporary tensors to free memory
            del ddd, www, zigzag, linear

        module._is_hf_initialized = True

    @staticmethod
    def _orthogonal_init(weight, gain=1.0):
        oringinal_dtype = weight.dtype
        weight = weight.float()
        nn.init.orthogonal_(weight, gain=gain)
        weight = weight.to(oringinal_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_params: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        v_first: torch.Tensor = None,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        batch_size, seq_len, _ = hidden_states.shape
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
            am = attention_mask.narrow(1, attention_mask.size(1) - seq_len, seq_len).unsqueeze(-1)

        if cache_params is not None and not isinstance(cache_params, RWKV7Cache):
            raise TypeError(
                "RWKV7Attention expects an RWKV7Cache; "
                f"got {type(cache_params).__name__}."
            )
        last_state = (
            cache_params.layers[self.layer_idx]
            if cache_params is not None
            else None
        )

        if attention_mask is not None:
            hidden_states = hidden_states.mul(am)

        # delta [batch_size, seq_len, hidden_size]
        # conv_cache [N, D]
        if last_state is None:
            conv_cache = None
            recurrent_state = None
        else:
            conv_cache = last_state.conv_state
            recurrent_state = last_state.recurrent_state

        delta, conv_state = token_shift(
            hidden_states, cu_seqlens, output_cache=True, cache=conv_cache,
        )
        xr, xw, xk, xv, xa, xg = addcmul_rwkv7(hidden_states, delta, self.x_r, self.x_w,
                                                     self.x_k, self.x_v, self.x_a, self.x_g)

        r = self.r_proj(xr)
        # Using bf16 for LoRA computation is numerically safe here because:
        # 1. After sigmoid activation:
        #    - Max absolute error (vs float32): 0.003
        #    - Mean absolute error: 0.0004
        # 2. Subsequent scaling by -0.6065 will further reduce relative error
        #    (error scales linearly with constant multiplication)
        # 3. Final compounded error remains within acceptable bounds for bf16 precision
        # Empirical observation confirms bf16 introduces no practical degradation
        w = -0.6065306597126334 * self.w_lora(xw).sigmoid()

        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        kk = F.normalize(
            rearrange(k * self.k_k, "b t (h d) -> b t h d", d=self.head_dim),
            dim=-1,
            p=2.0,
        )

        # Prefer addcmul over expanded form for numerical stability in bf16:
        # 1. Fused Multiply-Add (FMA) in addcmul reduces intermediate rounding:
        #    - Single op vs original 3 ops (mul, sub, mul)
        #    - 1 less intermediate value storage (bf16 write->read overhead)
        # 2. Mathematically equivalent to k*(1 + (a-1)*self.k_a)
        #    but with better precision preservation
        # 3. Particularly crucial for bf16 where intermediate values easily lose precision
        # 4. Pytorch method: k = k.addcmul(k * (a - 1), self.k_a)
        k = k_update_rwkv7(k, a, self.k_a)

        # dealing with left-padding
        if attention_mask is not None:
            v = v * am

        r, w, k, a = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim), (r, w, k, a))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        if self.training or seq_len >= 64:
            # if training, use chunk mode no matter how short the sequence is
            # launching the triton kernel for just one token will actually be slower
            o, recurrent_state = wkv_chunk(
                r=r,
                w=w,
                k=k,
                v=v,
                kk=kk,
                a=a,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                chunk_size=64,
            )
        else:
            o, recurrent_state = wkv_recurrent(
                r=r,
                w=w,
                k=k,
                v=v,
                kk=kk,
                a=a,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )

        if cache_params is not None and use_cache:
            cache_params.update(
                layer_idx=self.layer_idx,
                recurrent_state=recurrent_state,
                conv_state=conv_state,
                offset=seq_len,
            )

        o = self.g_norm(rearrange(o, "b t h d -> (b t) (h d)")).view(
            batch_size, seq_len, -1
        )
        o = output_correction(o, r, k, self.r_k, v, g)
        o = self.o_proj(o)

        return o, None, cache_params, v_first
    
class RWKV7FeedForward(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: int | None = None,
        intermediate_size: int | None = None,
        hidden_act: str = 'sqrelu',
        layer_idx: int = None,
        num_hidden_layers: int = None,
    ) -> RWKV7FeedForward:
        super().__init__()

        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio)
            intermediate_size = 32 * ((intermediate_size + 32 - 1) // 32)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.x_k = nn.Parameter(torch.zeros(hidden_size))

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)
        if hidden_act != "sqrelu":
            raise ValueError(
                f"This RWKV7 implementation only supports hidden_act='sqrelu', got {hidden_act!r}."
            )
        self.act_fn = sqrelu

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        try:
            from transformers.modeling_utils import _init_weights
        except ImportError:
            _init_weights = True
        if _init_weights:
            self.apply(self._initialize_weights)
        for name, module in self.named_modules():
            module._in_rwkv_module = True

    def _initialize_weights(self, module: nn.Module):
        if isinstance(module, RWKV7FeedForward):
            if module.key.weight is None or module.key.weight.is_meta:
                module._is_hf_initialized = True
                return
            with torch.no_grad():
                ratio_1_to_almost0 = 1.0 - (module.layer_idx / module.num_hidden_layers)  # 1 to ~0
                ddd = torch.ones(1, 1, module.hidden_size)
                for i in range(module.hidden_size):
                    ddd[0, 0, i] = i / module.hidden_size
                module.x_k.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0**4).squeeze()

            # Initialize key and value weights as in CMix_x070
            original_dtype = module.key.weight.dtype
            module.key.weight.data = nn.init.orthogonal_(module.key.weight.data.to(torch.float32)).to(original_dtype)
            module.value.weight.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_params: Cache | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        use_cache: bool | None = False,
        **kwargs,
    ) -> torch.Tensor:
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        if cache_params is not None:
            if not isinstance(cache_params, RWKV7Cache):
                raise TypeError(
                    "RWKV7FeedForward expects an RWKV7Cache; "
                    f"got {type(cache_params).__name__}."
                )
            delta, ffn_state = token_shift(
                x,
                cu_seqlens,
                cache=cache_params.layers[self.layer_idx].ffn_state,
                output_cache=True,
            )
        else:
            delta, ffn_state = token_shift(x, cu_seqlens, output_cache=True)
        if cache_params is not None and use_cache:
            cache_params.update(
                layer_idx=self.layer_idx,
                ffn_state=ffn_state,
                offset=0,
            )
        return self.value(self.act_fn(self.key(x.addcmul(delta, self.x_k)))), cache_params


class RWKV7Block(GradientCheckpointingLayer):

    def __init__(
        self,
        config: RWKV7Config,
        layer_idx: int,
    ) -> RWKV7Block:
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        if config.norm_first and layer_idx == 0:
            self.pre_norm = nn.LayerNorm(
                config.hidden_size,
                bias=config.norm_bias,
                eps=config.norm_eps,
            )
        self.attn_norm = nn.LayerNorm(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps,
        )
        if config.attn is not None and layer_idx in config.attn['layers']:
            # Optional hybrid-attention path; the converted RWKV7 checkpoints
            # use attn=None and never import this FLA layer module.  Use
            # importlib so transformers' static import checker does not make
            # the whole modeling file require FLA.
            from importlib import import_module

            attention_cls = import_module("fla.layers.attn").Attention

            self.attn = attention_cls(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                qkv_bias=config.attn['qkv_bias'],
                window_size=config.attn['window_size'],
                rope_theta=config.attn['rope_theta'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx,
            )
        else:
            self.attn = RWKV7Attention(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                decay_low_rank_dim=config.decay_low_rank_dim,
                gate_low_rank_dim=config.gate_low_rank_dim,
                a_low_rank_dim=config.a_low_rank_dim,
                v_low_rank_dim=config.v_low_rank_dim,
                norm_eps=config.norm_eps,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx,
                value_dim=config.value_dim[layer_idx],
                num_hidden_layers=config.num_hidden_layers,
            )
        self.ffn_norm = nn.LayerNorm(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps,
        )
        self.ffn = RWKV7FeedForward(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_idx=layer_idx,
            num_hidden_layers=config.num_hidden_layers,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache_params: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        v_first: torch.Tensor = None,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = self.pre_norm(hidden_states) if hasattr(self, 'pre_norm') else hidden_states
        hidden_states = self.attn_norm(residual)
        hidden_states, attentions, cache_params, v_first = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cache_params=cache_params,
            use_cache=use_cache,
            output_attentions=output_attentions,
            v_first=v_first,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states, cache_params = self.ffn(
            hidden_states,
            attention_mask,
            cache_params,
            cu_seqlens,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, cache_params, v_first)

        return outputs


class RWKV7PreTrainedModel(PreTrainedModel):

    config_class = RWKV7Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['RWKV7Block']
    _supports_cache_class = True
    _is_stateful = True
    # Automodel THD packing: the model consumes per-document lengths itself.
    _owns_packed_attention = True
    _skip_keys_device_placement = ["past_key_values", "cache_params"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    @torch.no_grad()
    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, nn.Embedding):
            # https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp/src/model.py#L396C12-L399C58
            if not getattr(module.weight, '_is_hf_initialized', False):
                scale = -1e-4
                nn.init.uniform_(module.weight, a=scale, b=-scale)
        elif isinstance(module, nn.Linear) and hasattr(self, 'lm_head') and module is self.lm_head:
            # https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp/src/model.py#L403
            if not getattr(module.weight, '_is_hf_initialized', False):
                if self.config.vocab_size > self.config.hidden_size:
                    scale = 0.5 * math.sqrt(self.config.vocab_size / self.config.hidden_size)
                else:
                    scale = 0.5
                original_dtype = module.weight.dtype
                module.weight.data = nn.init.orthogonal_(module.weight.data.to(torch.float32), gain=scale).to(original_dtype)
        # Init Attention parameters
        elif isinstance(module, (nn.Linear, nn.Conv1d)) and getattr(module, '_in_rwkv_module', False) is False:
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters') and getattr(module, '_in_rwkv_module', False) is False:
            module.reset_parameters()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class RWKV7Model(RWKV7PreTrainedModel):

    def __init__(self, config: RWKV7Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([RWKV7Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps,
        )

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Override the load_state_dict method to handle migration from version 1 to version 2.
        Handles hierarchical keys like 'model.layers.0.attn.x_x'.
        """
        # Collect all layer indices from the state_dict keys
        layer_indices = set()
        for key in state_dict.keys():
            if key.startswith("model.layers."):
                # Extract the layer index from the key
                try:
                    layer_idx = int(key.split(".")[2])  # Extract the number after 'model.layers.'
                    layer_indices.add(layer_idx)
                except ValueError:
                    # Skip keys that don't match the expected format
                    continue

        # Sort the layer indices to process them in order
        sorted_layer_indices = sorted(layer_indices)

        # Migration logic for each layer
        for layer_idx in sorted_layer_indices:
            layer_prefix = f"model.layers.{layer_idx}"
            attn_prefix = f"{layer_prefix}.attn"

            # Check if the layer contains the old 'x_x' parameter
            if f"{attn_prefix}.x_x" in state_dict:
                logger.info(f"Migrating weights for layer {layer_idx} from RWKV7Attention version 1 to version 2...")
                # Extract the x_x parameter
                x_x = state_dict[f"{attn_prefix}.x_x"]
                with torch.no_grad():
                    # Create new parameters for version 2
                    state_dict[f"{attn_prefix}.x_r"] = x_x[0].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_w"] = x_x[1].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_k"] = x_x[2].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_v"] = x_x[3].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_a"] = x_x[4].unsqueeze(0).unsqueeze(0)
                    state_dict[f"{attn_prefix}.x_g"] = x_x[5].unsqueeze(0).unsqueeze(0)

        # Call the parent method to load the modified state_dict
        try:
            super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            # If the parent method does not support `assign`, fall back to strict loading
            logger.warning(
                "`assign` parameter is not supported by the parent `load_state_dict` method. "
                "Falling back to default behavior.",
            )
            super().load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Cache | None = None,
        cache_params: Cache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        seq_lens: torch.LongTensor | None = None,
        seq_lens_padded: torch.LongTensor | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple | RWKV7ModelOutput:
        if output_attentions:
            warnings.warn("`RWKV7Model` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds
        if cu_seqlens is None:
            length_source = seq_lens_padded if seq_lens_padded is not None else seq_lens
            if length_source is not None:
                cu_seqlens = _seq_lens_to_cu_seqlens(length_source).to(
                    hidden_states.device
                )
        if cu_seqlens is not None:
            use_cache = False
        if cu_seqlens is not None and use_cache:
            raise ValueError(
                "use_cache=True is not supported for packed cu_seqlens input; "
                "packed training should use use_cache=False."
            )
        if cu_seqlens is not None and hidden_states.shape[0] != 1:
            raise ValueError(
                "Packed sequences via cu_seqlens require batch_size=1; "
                f"got batch_size={hidden_states.shape[0]}."
            )

        if cache_params is not None and past_key_values is not None:
            if cache_params is not past_key_values:
                raise ValueError(
                    "Pass either `cache_params` or `past_key_values`, not two "
                    "different cache objects."
                )
        if cache_params is None:
            cache_params = past_key_values
        if cache_params is not None and not isinstance(cache_params, RWKV7Cache):
            raise TypeError(
                "RWKV7Model expects an RWKV7Cache; "
                f"got {type(cache_params).__name__}."
            )
        if use_cache and cache_params is None:
            cache_params = RWKV7Cache(self.config)

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        v_first = torch.zeros_like(hidden_states)
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, attentions, cache_params, v_first = layer(
                hidden_states,
                attention_mask=attention_mask,
                cache_params=cache_params,
                use_cache=use_cache,
                output_attentions=output_attentions,
                v_first=v_first,
                cu_seqlens=cu_seqlens,
                **kwargs,
            )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                item
                for item in [hidden_states, cache_params, all_hidden_states, all_attns]
                if item is not None
            )
        return RWKV7ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=cache_params,
            cache_params=cache_params,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class RWKV7ForCausalLM(RWKV7PreTrainedModel, GenerationMixin):

    # The RWKV7 checkpoints store a separate LM head; do not tie it to the
    # embedding matrix.
    _tied_weights_keys: list[str] = []

    def __init__(self, config):
        super().__init__(config)
        self.model = RWKV7Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # RWKV7 uses its own recurrent cache, not key/value tensors.
        return False

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = True,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | None = None,
        **kwargs,
    ):
        cache_params = kwargs.pop("cache_params", None)
        if cache_params is not None and past_key_values is not None:
            if cache_params is not past_key_values:
                raise ValueError(
                    "Pass either `cache_params` or `past_key_values`, not two "
                    "different cache objects."
                )
        cache = cache_params if cache_params is not None else past_key_values
        if cache is not None and not isinstance(cache, RWKV7Cache):
            raise TypeError(
                "RWKV7 generation expects an RWKV7Cache; "
                f"got {type(cache).__name__}."
            )

        use_cache = True if use_cache is None else bool(use_cache)
        if use_cache and cache is None:
            cache = RWKV7Cache(self.config)
        if not use_cache:
            cache = None

        # Decode steps only need the newly generated token.  Prefer
        # cache_position because it also works with left-padded prompts.
        if (
            input_ids is not None
            and cache is not None
            and cache.get_seq_length() > 0
        ):
            if cache_position is not None and input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]
            elif input_ids.shape[1] > 1:
                input_ids = input_ids[:, -1:]

        model_inputs: dict[str, object] = {
            "input_ids": input_ids,
            "cache_params": cache,
            "use_cache": use_cache,
        }
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if inputs_embeds is not None and (cache is None or cache.get_seq_length() == 0):
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs["input_ids"] = None
        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_params: Cache | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        seq_lens: torch.LongTensor | None = None,
        seq_lens_padded: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        shift_labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        **kwargs: Unpack[dict],
    ) -> tuple | RWKV7CausalLMOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if cu_seqlens is None:
            length_source = seq_lens_padded if seq_lens_padded is not None else seq_lens
            if length_source is not None:
                token_device = (
                    input_ids.device
                    if input_ids is not None
                    else inputs_embeds.device if inputs_embeds is not None else None
                )
                if token_device is None:
                    raise ValueError(
                        "seq_lens was provided but neither input_ids nor inputs_embeds is set."
                    )
                cu_seqlens = _seq_lens_to_cu_seqlens(length_source).to(token_device)
        if cu_seqlens is not None:
            # Packed sequences are stateless: never create/use a cache.
            use_cache = False
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_params=cache_params,
            cu_seqlens=cu_seqlens,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        if return_dict:
            cache = outputs.cache_params
            all_hidden_states = outputs.hidden_states
            all_attentions = outputs.attentions
        else:
            cache = outputs[1] if len(outputs) > 1 else None
            all_hidden_states = outputs[2] if len(outputs) > 2 else None
            all_attentions = outputs[3] if len(outputs) > 3 else None

        if (labels is not None or shift_labels is not None) and logits_to_keep:
            raise ValueError(
                "`labels`/`shift_labels` cannot be combined with "
                "`logits_to_keep > 0`."
            )

        logits = self.lm_head(
            hidden_states
            if not logits_to_keep
            else hidden_states[:, -logits_to_keep:]
        )
        loss = None
        if labels is not None or shift_labels is not None:
            criterion = nn.CrossEntropyLoss()

            # shift_labels: See https://github.com/huggingface/transformers/pull/36607/files.
            if shift_labels is None:
                shift_labels = torch.cat(
                    (
                        labels[..., 1:],
                        torch.full_like(labels[:, :1], criterion.ignore_index),
                    ),
                    1,
                )
                if cu_seqlens is not None:
                    # Packed sequences: the last token of each segment must
                    # not predict the first token of the next segment.
                    if labels.shape[0] != 1:
                        raise ValueError(
                            "Packed training with cu_seqlens requires batch_size=1."
                        )
                    if seq_lens is not None:
                        lengths = torch.as_tensor(
                            seq_lens,
                            device=shift_labels.device,
                            dtype=torch.long,
                        ).reshape(-1)
                        lengths = lengths[lengths > 0]
                        internal_ends = lengths.cumsum(0)[:-1]
                    else:
                        internal_ends = torch.as_tensor(
                            cu_seqlens,
                            device=shift_labels.device,
                            dtype=torch.long,
                        )[1:-1]
                    valid = (internal_ends > 0) & (
                        internal_ends <= shift_labels.shape[1]
                    )
                    if valid.any():
                        boundary_idx = (internal_ends[valid] - 1).unsqueeze(0)
                        shift_labels.scatter_(
                            1,
                            boundary_idx.expand(shift_labels.shape[0], -1),
                            criterion.ignore_index,
                        )
            shift_labels = shift_labels.to(hidden_states.device)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                shift_labels.reshape(-1),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return RWKV7CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=cache,
            cache_params=cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
