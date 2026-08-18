import mlx.core as mx


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

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:

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
