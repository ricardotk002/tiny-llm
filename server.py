from mlx_lm import load
from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx.core as mx
import argparse
from typing import Any, Callable
import math

def softmax(x: mx.array, axis: int) -> mx.array:
  z = mx.exp(x - mx.max(x, axis=axis, keepdims=True)) 
  return z / mx.sum(z, axis=axis, keepdims=True)

def dequantize_linear(mx_layer: Any) -> mx.array:
  w = mx.dequantize(mx_layer.weight, mx_layer.scales, mx_layer.biases, mx_layer.group_size, mx_layer.bits)
  return w.astype(mx.bfloat16)

def linear(x: mx.array, w: mx.array, bias: mx.array | None = None) -> mx.array:
  return x @ w.T if bias is None else x @ w.T + bias

def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
  out = mx.zeros((L, S))
  a = mx.arange(L)[:, None]
  b = mx.arange(S)[None, :]
  out[a < b - (S - L)] = -mx.inf
  return out

def silu(x: mx.array) -> mx.array:
  z = mx.exp(-abs(x)) # Prevents exp(-x) to become exp(large positive)
  return mx.where(x >= 0, x / (1 + z), x * z / (1 + z))

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

class RoPE:
  def __init__(
    self,
    dims: int,
    seq_len: int,
    base: int = 10000,
    traditional: bool = False,
  ):
    self.dims = dims
    self.seq_len = seq_len
    self.base = base
    self.traditional = traditional
    self.theta = 1 / (base ** (mx.arange(0, self.dims, 2) / self.dims))
    self.seq_idx = mx.arange(seq_len)
    self.idx_theta = self.seq_idx[:, None] * self.theta[None, :] # einsum
    self.cache = mx.stack([self.idx_theta.cos(), self.idx_theta.sin()], axis=-1)

  def __call__(self, x: mx.array, offset: list[slice] | slice | None = None) -> mx.array:
    seq_len = x.shape[1]

    if offset is not None:
      if isinstance(offset, slice):
        cache = self.cache[offset]
      else:
        cache = self.cache[:offset]
    else:
      cache = self.cache[:seq_len]

    # TODO: This should be for traditional RoPE only
    x_shaped = mx.reshape(x, (*x.shape[:-1], -1, 2))
    cache_shaped = mx.reshape(cache, (-1, x_shaped.shape[1], 1, x_shaped.shape[3], 2))

    if self.traditional:
      x_out = mx.stack([
        # cos = 0, sin = 1
        x_shaped[..., 0] * cache_shaped[..., 0] - x_shaped[..., 1] * cache_shaped[..., 1],
        x_shaped[..., 1] * cache_shaped[..., 0] + x_shaped[..., 0] * cache_shaped[..., 1]
      ], axis=-1)
    else:
      x_1 = x[..., : self.dims // 2]
      x_2 = x[..., self.dims // 2 :]

      out_1 = x_1 * cache_shaped[..., 0] - x_2 * cache_shaped[..., 1]
      out_2 = x_1 * cache_shaped[..., 1] + x_2 * cache_shaped[..., 0]

      return mx.concatenate([out_1, out_2], axis=-1)

    return x_out.flatten(3)

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

    q = RMSNorm(self.head_dim, self.q_norm, eps=self.rms_norm_eps)(q)
    k = RMSNorm(self.head_dim, self.k_norm, eps=self.rms_norm_eps)(k)

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

class Qwen3Model:
  def __init__(self, mlx_model: Any):
    self.mlx_model = mlx_model
    self.vocab_size = mlx_model.args.vocab_size
    self.hidden_size = mlx_model.args.hidden_size
    self.num_attention_heads = mlx_model.args.num_attention_heads
    self.num_kv_heads = mlx_model.args.num_key_value_heads
    self.head_dim = mlx_model.args.head_dim
    self.intermediate_size = mlx_model.args.intermediate_size
    self.rms_norm_eps = mlx_model.args.rms_norm_eps
    self.theta = mlx_model.args.rope_theta

  def __call__(self, inputs: mx.array) -> mx.array:
    x = Embedding(
      vocab_size=self.vocab_size,
      embedding_dim=self.hidden_size,
      weight=dequantize_linear(self.mlx_model.model.embed_tokens)
    )(inputs)

    for layer in self.mlx_model.layers:
      x = Qwen3TransformerBlock(
        num_attention_heads=self.num_attention_heads,
        num_kv_heads=self.num_kv_heads,
        hidden_size=self.hidden_size,
        head_dim=self.head_dim,
        intermediate_size=self.intermediate_size,
        rms_norm_eps=self.rms_norm_eps,
        wq=dequantize_linear(layer.self_attn.q_proj),
        wk=dequantize_linear(layer.self_attn.k_proj),
        wv=dequantize_linear(layer.self_attn.v_proj),
        wo=dequantize_linear(layer.self_attn.o_proj),
        q_norm=layer.self_attn.q_norm.weight,
        k_norm=layer.self_attn.k_norm.weight,
        w_gate=dequantize_linear(layer.mlp.gate_proj),
        w_up=dequantize_linear(layer.mlp.up_proj),
        w_down=dequantize_linear(layer.mlp.down_proj),
        w_input_layernorm=layer.input_layernorm.weight,
        w_post_attention_layernorm=layer.post_attention_layernorm.weight,
        theta=self.theta
      )(x, mask="causal")

    x = RMSNorm(self.head_dim, self.mlx_model.model.norm.weight, eps=self.rms_norm_eps)(x)

    x = Embedding(
      vocab_size=self.vocab_size,
      embedding_dim=self.hidden_size,
      weight=dequantize_linear(self.mlx_model.model.embed_tokens)
    ).as_linear(x)

    return x

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

  def __call__(self, x: mx.array, mask: mx.array | str | None = None) -> mx.array:
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

class Embedding:
  def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.weight = weight

  def __call__(self, x: mx.array) -> mx.array:
    return self.weight[x]

  def as_linear(self, x: mx.array) -> mx.array:
    return linear(x, self.weight)

class RMSNorm:
  def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
    self.dim = dim
    self.weight = weight
    self.eps = eps

  def __call__(self, x: mx.array) -> mx.array:
    x = x.astype(mx.float32)
    return ((x / mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + self.eps))) * self.weight

def generate(
  model: Qwen3Model,
  tokenizer: TokenizerWrapper,
  prompt: str,
  sampler: Callable[[mx.array], mx.array] | None,
) -> None:
  def _step(model, y):
    y = mx.expand_dims(mx.array(y), 0)
    out = model(y)
    logits = out[:, -1, :] # only last token's logits
    return sampler(logits)

  tokens = tokenizer.encode(prompt)
  detokenizer = tokenizer.detokenizer # it creates a new detokenizer on each call, weird

  tok = _step(model, tokens) # prefill
  while tok.item() != tokenizer.eos_token_id:
    detokenizer.add_token(tok.item())
    print(detokenizer.last_segment, end="", flush=True)
    tokens.append(tok.item())
    tok = _step(model, tokens)

  mx.clear_cache()

def make_sampler(temp: float, top_p: float | None, top_k: int | None):
  def sample(logprobs: mx.array):
    if temp == 0:
      return mx.argmax(logprobs, axis=-1)

    if temp > 0:
      if top_k is not None:
        top = mx.argpartition(logprobs, -top_k, axis=-1)[:,-top_k:]
        masked_logits = mx.full_like(logprobs, -mx.inf)
        masked_logits[:,top] = logprobs[:,top]

        return mx.random.categorical(masked_logits / temp)

      return mx.random.categorical(logprobs / temp)

  return sample

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default="qwen3-0.6b")
  parser.add_argument('--prompt', type=str, default="Give me a short introduction to large language model.")
  
  args = parser.parse_args()
  
  MODEL_NAMES = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B-MLX-4bit",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B-MLX-4bit",
  }
  
  model_name = MODEL_NAMES.get(args.model)
  model, tokenizer = load(model_name)

  with mx.stream(mx.gpu):
    qwen3 = Qwen3Model(model)
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
  
    try:
      while True:
        user_input = input("\n> ")
        messages.append({ "role": "user", "content": user_input })
  
        sampler = make_sampler(0.75, top_p=None, top_k=10)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        generate(qwen3, tokenizer, prompt, sampler=sampler)
  
        if user_input.strip() == "exit":
          break
    finally:
      mx.clear_cache()
