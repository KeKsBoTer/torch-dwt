import torch
import torch.nn.functional as F


@torch.jit.script
def _idwt1d(
    lo: torch.Tensor, hi: torch.Tensor, lo_hi: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """performs a 1d idwt on the defined dimension

    Args:
        lo (torch.Tensor): 5d low (average) coefs of shape [N,C,D,H,W]
        hi (torch.Tensor): 5d hi (detail) coefs of shape [N,C,D,H,W]
        lo_hi (torch.Tensor): lo,hi pass filters of shape [2,K]
        dim (int, optional): dimension to apply the idwt to . Defaults to -1.

    Returns:
        torch.Tensor: reconstructed tensor of shape [N,C,D_out,H_out,W_out] (e.g. H_out = 2*H if dim==-1)
    """
    dim = dim % 5

    groups = lo.shape[1]
    filter_c = (
        lo_hi[:, None, None, None, None, :]
        .repeat(1, groups, 1, 1, 1, 1)
        .swapaxes(5, dim + 1)  # swap filter to dwt dim
    )

    # stride of 2 for dwt dim
    stride = [1, 1, 1]
    stride[dim - 2] = 2
    padding = [0, 0, 0]
    padding[dim - 2] = lo_hi.shape[-1] - 2

    a_coefs = F.conv_transpose3d(
        lo, filter_c[0], stride=stride, padding=padding, groups=groups
    )
    d_coefs = F.conv_transpose3d(
        hi, filter_c[1], stride=stride, padding=padding, groups=groups
    )
    return a_coefs + d_coefs


@torch.jit.script
def _dwt1d(x: torch.Tensor, lo_hi: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """performs a 1d dwt on the defined dimension

    Args:
        x (torch.Tensor): 4d tensor of shape [N,C,D,H,W]
        lo_hi (torch.Tensor): low and highpass filter (shape [2,K])
        dim (int, optional): dimension to apply the dwt to . Defaults to -1.

    Returns:
        torch.Tensor: dwt coefs of shape [N,2,C,D_out,H_out,W_out]. The average and detail coefs are concatenated in the channels
    """
    dim = dim % 5
    groups = x.shape[1]
    # repeat filter to match number of channels
    filter_c = lo_hi[:, None, None, None, :].repeat(groups, 1, 1, 1, 1).swapaxes(4, dim)

    if x.shape[dim] % 2 != 0:
        # pad dwt dimension to multiple of two
        pad = [0] * 6
        pad[(4 - dim) * 2 + 1] = 1
        x = F.pad(x, pad)

    # stride of 2 for dwt dim
    stride = [1, 1, 1]
    stride[dim - 2] = 2

    padding = [0, 0, 0]
    padding[dim - 2] = lo_hi.shape[-1] - 2

    filtered = F.conv3d(x, filter_c, stride=stride, padding=padding, groups=groups)
    return filtered.reshape(
        filtered.shape[0],
        groups,
        2,
        filtered.shape[2],
        filtered.shape[3],
        filtered.shape[4],
    ).swapaxes(1, 2)


@torch.jit.script
def _dwt2(x: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
    lh = _dwt1d(x[:, :, None, :, :], lohi, -1)
    y = _dwt1d(lh.flatten(1, 2), lohi, -2).squeeze(-3)

    # reorder coefs to match pywt ordering
    return y.reshape(x.shape[0], 4, x.shape[1], y.shape[-2], y.shape[-1])


@torch.jit.script
def _idwt2(x: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
    ll, hl, lh, hh = x.swapaxes(1, 2).unsqueeze(-3).unbind(2)

    lo = _idwt1d(ll, lh, lohi, dim=-2)

    hi = _idwt1d(hl, hh, lohi, dim=-2)
    y = _idwt1d(lo, hi, lohi, dim=-1)
    return y.squeeze(-3)


@torch.jit.script
def _dwt3(x: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
    x_c = _dwt1d(x, lohi, -1)
    y_c = _dwt1d(x_c.flatten(1, 2), lohi, -2)
    z_c = _dwt1d(y_c.flatten(1, 2), lohi, -3)
    return z_c.reshape(
        z_c.shape[0], 8, x.shape[1], z_c.shape[3], z_c.shape[4], z_c.shape[5]
    )


@torch.jit.script
def _idwt3(x: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
    lll, hll, lhl, hhl, llh, hlh, lhh, hhh = x.unbind(1)

    ll = _idwt1d(lll, llh, lohi, dim=-3)
    hl = _idwt1d(hll, hlh, lohi, dim=-3)

    lh = _idwt1d(lhl, lhh, lohi, dim=-3)
    hh = _idwt1d(hhl, hhh, lohi, dim=-3)

    l = _idwt1d(ll, lh, lohi, dim=-2)
    h = _idwt1d(hl, hh, lohi, dim=-2)
    y = _idwt1d(l, h, lohi, dim=-1)
    return y
