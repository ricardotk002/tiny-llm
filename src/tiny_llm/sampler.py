import mlx.core as mx
import copy


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
