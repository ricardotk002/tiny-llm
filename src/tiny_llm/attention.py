import mlx.core as mx
from .basics import softmax, linear
import math

def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    if scale is None:
        scale = 1 / math.sqrt(key.shape[-1])

    scores = (query @ mx.transpose(key, axes=(*range(key.ndim - 2), -1, -2))) * scale

    if mask is not None:
        scores = scores + mask
    
    return softmax(scores, axis=-1) @ value

class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.head_dim = hidden_size // num_heads

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        q = linear(query, self.wq)
        k = linear(key, self.wk)
        v = linear(value, self.wv)

        q = mx.reshape(q, (query.shape[0], query.shape[1], self.num_heads, self.head_dim))
        k = mx.reshape(k, (key.shape[0], key.shape[1], self.num_heads, self.head_dim))
        v = mx.reshape(v, (value.shape[0], value.shape[1], self.num_heads, self.head_dim))

        q = q.swapaxes(-3,-2)
        k = k.swapaxes(-3,-2)
        v = v.swapaxes(-3,-2)

        attn = scaled_dot_product_attention_simple(q, k, v, None, mask)

        attn = attn.swapaxes(-3,-2)
        attn = mx.reshape(attn, (value.shape[0], value.shape[1], self.num_heads * self.head_dim))

        return linear(attn, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    o = mx.zeros((L, S))
    a = mx.arange(L)[:, None]
    b = mx.arange(S)[None, :]
    o[a < b - (S - L)] = -mx.inf
    return o

def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    H_q = query.shape[-3] # query heads
    H = key.shape[-3] # kv heads to be shared

    if scale is None:
        scale = 1 / math.sqrt(key.shape[-1])

    query = mx.reshape(query, (*query.shape[:-3], H, -1, query.shape[-2], query.shape[-1]))
    key = mx.expand_dims(key, axis=-3)
    value = mx.expand_dims(value, axis=-3)

    scores = (query @ mx.transpose(key, axes=(*range(key.ndim - 2), -1, -2))) * scale

    if mask is not None and isinstance(mask, mx.array):
        mask = mx.reshape(mask, (*mask.shape[:-3], H, -1, mask.shape[-2], mask.shape[-1]))
        scores = scores + mask

    if mask is not None and mask == "causal":
        mask = causal_mask(query.shape[-2], key.shape[-2], dtype=scores.dtype)
        scores = scores + mask

    attn = softmax(scores, axis=-1) @ value

    return mx.reshape(attn, (*attn.shape[:-4], H_q, attn.shape[-2], attn.shape[-1]))

def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass
