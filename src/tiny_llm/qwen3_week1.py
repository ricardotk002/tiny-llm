import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rms_norm_eps = rms_norm_eps

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        L = x.shape[-2]

        q = linear(x, self.wq)
        k = linear(x, self.wk)
        v = linear(x, self.wv)

        q = mx.reshape(q, (*q.shape[:-1], self.num_heads, -1))
        k = mx.reshape(k, (*k.shape[:-1], self.num_kv_heads, -1))
        v = mx.reshape(v, (*v.shape[:-1], self.num_kv_heads, -1))

        q = mx.fast.rms_norm(q, self.q_norm, eps=self.rms_norm_eps)
        k = mx.fast.rms_norm(k, self.k_norm, eps=self.rms_norm_eps)

        q = RoPE(self.head_dim, self.max_seq_len, self.theta)(q, slice(0, L))
        k = RoPE(self.head_dim, self.max_seq_len, self.theta)(k, slice(0, L))

        q = q.swapaxes(-3,-2)
        k = k.swapaxes(-3,-2)
        v = v.swapaxes(-3,-2)

        x = scaled_dot_product_attention_grouped(q, k, v, None, mask)

        x = x.swapaxes(-3,-2)
        x = mx.reshape(x, (*x.shape[:-2], -1))


        return linear(x, self.wo)

class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        gate = silu(linear(x, self.w_gate))
        up = linear(x, self.w_up)
        return linear(gate * up, self.w_down)


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.w_input_layernorm = w_input_layernorm
        self.w_post_attention_layernorm = w_post_attention_layernorm
        self.max_seq_len = max_seq_len
        self.theta = theta

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        x1 = RMSNorm(self.head_dim, self.w_input_layernorm, self.rms_norm_eps)(x)
        x1 = Qwen3MultiHeadAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            wq=self.wq,
            wk=self.wk,
            wv=self.wv,
            wo=self.wo,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            max_seq_len=self.max_seq_len,
            theta=self.theta,
            rms_norm_eps=self.rms_norm_eps
        )(x1, mask)
        x = x + x1 # add residual
        x2 = RMSNorm(self.head_dim, self.w_post_attention_layernorm, self.rms_norm_eps)(x)
        x2 = Qwen3MLP(
            dim=self.head_dim,
            hidden_dim=self.hidden_size,
            w_gate=self.w_gate,
            w_up=self.w_up,
            w_down=self.w_down
        )(x2)
        x = x + x2 # add residual

        return x


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
