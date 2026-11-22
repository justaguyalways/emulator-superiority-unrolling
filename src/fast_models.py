"""Channel-last evaluation of the existing 1D CNN/UNet, with unchanged parameters.

On this CPU, small channel-first convolutions are expensive. Express their
periodic stencils as matrix products. This changes floating-point summation
order, not the architecture, padding, initialization, or training objective.
Run validate_extension.py before using this backend for experiments.
"""
import equinox as eqx
import jax
import jax.numpy as jnp


def conv(c, x):
    """Periodic cross-correlation; x is (points, channels)."""
    if (c.num_spatial_dims != 1 or c.groups != 1 or
        (c.padding_mode != "circular" and c.kernel_size != (1,))):
        raise ValueError("Fast backend only supports ungrouped periodic 1D convolutions")
    k, d, s = c.kernel_size[0], c.dilation[0], c.stride[0]
    left, right = c.padding[0]
    if left + right != d * (k - 1):
        raise ValueError("Expected same padding")
    patches = jnp.concatenate([jnp.roll(x, left - i * d, axis=0)[::s]
                               for i in range(k)], axis=-1)
    weight = c.weight.transpose(0, 2, 1).reshape(c.out_channels, -1)
    out = patches @ weight.T
    return out + c.bias.T if c.use_bias else out


def upconv(c, x):
    if (c.kernel_size, c.stride, c.dilation, c.padding, c.output_padding,
        c.groups, c.padding_mode) != ((3,), (2,), (1,), ((1, 1),), (1,), 1, "circular"):
        raise ValueError("Unsupported transposed convolution configuration")
    expanded = jnp.stack((x, jnp.zeros_like(x)), axis=1).reshape(2 * x.shape[0], -1)
    patches = jnp.concatenate([jnp.roll(expanded, 1 - i, axis=0)
                               for i in range(3)], axis=-1)
    weight = c.weight.transpose(0, 2, 1).reshape(c.out_channels, -1)
    out = patches @ weight.T
    return out + c.bias.T if c.use_bias else out


def double_block(b, x):
    x = b.activation(b.norm_1(conv(b.conv_1, x).T).T)
    return b.activation(b.norm_2(conv(b.conv_2, x).T).T)


class FastModel(eqx.Module):
    model: eqx.Module
    kind: str = eqx.field(static=True)

    def __call__(self, x):
        m = self.model
        x = x.T
        if self.kind == "conv":
            for c in m.layers[:-1]:
                x = m.activation(conv(c, x))
            return m.final_activation(conv(m.layers[-1], x).T)
        if self.kind != "unet":
            raise ValueError(self.kind)
        x = double_block(m.lifting, x)
        skips = []
        for down, left in zip(m.down_sampling_blocks, m.left_arc_blocks):
            skips.append(x)
            x = conv(down, x)
            for b in left:
                x = double_block(b, x)
        for up, right in zip(reversed(m.up_sampling_blocks), reversed(m.right_arc_blocks)):
            x = upconv(up, x)
            x = jnp.concatenate((skips.pop(), x), axis=-1)
            for b in right:
                x = double_block(b, x)
        return conv(m.projection, x).T


def accelerate(model, network):
    kind = network.split(";")[0].lower()
    return FastModel(model, kind) if kind in ("conv", "unet") else model
