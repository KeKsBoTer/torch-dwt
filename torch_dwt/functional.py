from typing import Any
import torch
from pywt import Wavelet
import torch.nn.functional as F
from .lowlevel import _dwt1d, _idwt1d, _dwt2, _idwt2, _dwt3, _idwt3


def _to_wavelet_coefs(wavelet: str | torch.Tensor | Wavelet) -> torch.Tensor:
    match wavelet:
        case str():
            return torch.tensor(Wavelet(wavelet).filter_bank)[2:]
        case torch.Tensor():
            return wavelet
        case Wavelet():
            return torch.tensor(wavelet.filter_bank)[2:]
        case _:
            raise Exception("")


def dwt(x: torch.Tensor, wavelet: str | torch.Tensor | Wavelet) -> torch.Tensor:
    """performs the 1D discrete wavelet transform

    Args:
        x (torch.Tensor): input tensor shape [N,C,W]
        wavelet (str | torch.Tensor | Wavelet): wavelet (if tensor [2,C] (lo,hi filter))

    Raises:
        Exception: cannot handle wavelet type

    Returns:
        torch.Tensor: average, detail coefs of shape [N,2,C,W//2]
    """
    filter = _to_wavelet_coefs(wavelet).to(x.device)
    result = _dwt1d(x[:, :, None, None, :], filter, dim=-1)
    return result.reshape(x.shape[0], 2, x.shape[1], -1)


def idwt(x: torch.Tensor, wavelet: str | torch.Tensor | Wavelet) -> torch.Tensor:
    """performs the 1D inverse discrete wavelet transform

    Args:
        x (torch.Tensor): [N,2,C,W] average and detail coefs
        wavelet (str | torch.Tensor | Wavelet): wavelet

    Returns:
        torch.Tensor: reconstructed tensor
    """
    filter = _to_wavelet_coefs(wavelet).to(x.device)
    result = _idwt1d(
        x[:, 0, :, None, None, :], x[:, 1, :, None, None, :], filter, dim=-1
    )
    return result.squeeze(2).squeeze(2)


def dwt2(x: torch.Tensor, wavelet: str | torch.Tensor | Wavelet) -> torch.Tensor:
    """performs the 2D discrete wavelet transform

    Args:
        x (torch.Tensor): [N,C,H,W] data
        wavelet (str | torch.Tensor | Wavelet): wavelet

    Returns:
        torch.Tensor: dwt coefs of shape [M,4,C,H_out,W_out] \\
            (coef oder: cA,cV,cW,cD)
    """
    filter = _to_wavelet_coefs(wavelet).to(x.device)
    return _dwt2(x, filter)


def idwt2(x: torch.Tensor, wavelet: str | torch.Tensor | Wavelet) -> torch.Tensor:
    """performs the 2D inverse discrete wavelet transform

    Args:
        x (torch.Tensor): [N,4,C,H,W] average and detail coefs
        wavelet (str | torch.Tensor | Wavelet): wavelet

    Returns:
        torch.Tensor: reconstructed tensor
    """
    lohi = _to_wavelet_coefs(wavelet).to(x.device)
    return _idwt2(x, lohi)


def dwt3(x: torch.Tensor, wavelet: str | torch.Tensor | Wavelet) -> torch.Tensor:
    """performs the 2D discrete wavelet transform

    Args:
        x (torch.Tensor): [N,C,D,H,W] data
        wavelet (str | torch.Tensor | Wavelet): wavelet

    Returns:
        torch.Tensor: dwt coefs of shape [M,4*C,D_out,H_out,W_out] same order/shape as dtwn in pywt
    """
    filter = _to_wavelet_coefs(wavelet).to(x.device)
    return _dwt3(x, filter)


def idwt3(x: torch.Tensor, wavelet: str | torch.Tensor | Wavelet) -> torch.Tensor:
    """performs the 3D inverse discrete wavelet transform

    Args:
        x (torch.Tensor): [N,8,C,D,H,W] average and detail coefs
        wavelet (str | torch.Tensor | Wavelet): wavelet

    Returns:
        torch.Tensor: reconstructed tensor
    """
    lohi = _to_wavelet_coefs(wavelet).to(x.device)
    return _idwt3(x, lohi)


@torch.jit.script
def dwt3_2lvl(x: torch.Tensor, lohi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """performs a two level 3d dwt

    Args:
        x (torch.Tensor): data of shape [N,C,D,H,W]
        lohi (torch.Tensor): wavlet coefs of shape [2,K]

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 1lvl coefs of shape [N,7,C,D_1,H_1,W_1] \\
            2lvl coefs of shape [N,8,C,D_2,H_2,W_2]
    """
    coefs1 = _dwt3(x, lohi)
    coefs2 = _dwt3(coefs1[:, -1], lohi)
    return (coefs1[:, :-1], coefs2)


@torch.jit.script
def idwt3_2lvl(x1: torch.Tensor, x2: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
    """performs the inverse 2 level 3d dwt

    Args:
        x1 (torch.Tensor): level one coefs of shape [N,7,C,D_1,H_1,W_1]
        x2 (torch.Tensor): level two coefs of shape [N,8,C,D_2,H_2,W_2]
        lohi (torch.Tensor):  wavlet coefs of shape [2,K]

    Returns:
        torch.Tensor: reconstructed tensor of shape [N,C,D,H,W]
    """
    y_w = _idwt3(x2, lohi)
    # remove potential padding
    s = x1.shape[-3:]
    c2 = torch.cat([x1, y_w[:, None, :, : s[0], : s[1], : s[2]]], 1)
    return _idwt3(c2, lohi)


class DWT1D(torch.autograd.Function):
    """Performs the 1d dwt with a custom backward pass that uses less memory

    Args:
        x (torch.Tensor): input tensor of shape [N,C,W]
        lohi (torch.Tensor): filter bank of shape [2,K]

    Returns:
        torch.Tensor: average, detail coefs of shape [N,2,C,W//2]
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lohi)
        result = _dwt1d(x[:, :, None, None, :], lohi, dim=-1)
        return result.reshape(x.shape[0], 2, x.shape[1], -1)

    @staticmethod
    def backward(ctx: Any, dx0: torch.Tensor) -> torch.Tensor:
        dx = None
        if ctx.needs_input_grad[0]:
            (lohi,) = ctx.saved_tensors
            dx = _idwt1d(
                dx0[:, 0, :, None, None, :], dx0[:, 1, :, None, None, :], lohi
            )[:, :, 0, 0]
        return dx, None


class IDWT1D(torch.autograd.Function):
    """Performs the 1d inverse dwt with a custom backward pass that uses less memory

    Args:
        x (torch.Tensor): average and details coefs of shape [N,2,C,W]
        lohi (torch.Tensor): filter bank of shape [2,K]

    Returns:
        torch.Tensor: average, detail coefs of shape [N,C,W*2]
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lohi)
        result = _idwt1d(
            x[:, 0, :, None, None, :], x[:, 1, :, None, None, :], lohi, dim=-1
        )
        return result.squeeze(2).squeeze(2)

    @staticmethod
    def backward(ctx: Any, dx0: torch.Tensor) -> torch.Tensor:
        dx = None
        if ctx.needs_input_grad[0]:
            (lohi,) = ctx.saved_tensors

            dx = _dwt1d(dx0[:, :, None, None, :], lohi, -1)[:, :, :, 0, 0]

        return dx, None


class DWT2D(torch.autograd.Function):
    """Performs the 2d dwt with a custom backward pass that uses less memory

    Args:
        x (torch.Tensor): input tensor of shape [N,C,H,W]
        lohi (torch.Tensor): filter bank of shape [2,K]

    Returns:
        torch.Tensor: average, detail coefs of shape [N,4,C,H_out,W_out]
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lohi)
        return _dwt2(x, lohi)

    @staticmethod
    def backward(ctx: Any, dx0: torch.Tensor) -> torch.Tensor:
        dx = None
        if ctx.needs_input_grad[0]:
            (lohi,) = ctx.saved_tensors
            dx = _idwt2(dx0, lohi)
        return dx, None


class IDWT2D(torch.autograd.Function):
    """Performs the 2d inverse dwt with a custom backward pass that uses less memory

    Args:
        x (torch.Tensor): average and details coefs of shape [N,4,C,H,W]
        lohi (torch.Tensor): filter bank of shape [2,K]

    Returns:
        torch.Tensor: average, detail coefs of shape [N,C,H_out,W_out]
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lohi)
        return _idwt2(x, lohi)

    @staticmethod
    def backward(ctx: Any, dx0: torch.Tensor) -> torch.Tensor:
        dx = None
        if ctx.needs_input_grad[0]:
            (lohi,) = ctx.saved_tensors

            dx = _dwt2(dx0, lohi)

        return dx, None


class DWT3D(torch.autograd.Function):
    """Performs the 3d dwt with a custom backward pass that uses less memory

    Args:
        x (torch.Tensor): input tensor of shape [N,C,D,H,W]
        lohi (torch.Tensor): filter bank of shape [2,K]

    Returns:
        torch.Tensor: average, detail coefs of shape [N,8,C,D_out,H_out,W_out]
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lohi)
        return _dwt3(x, lohi)

    @staticmethod
    def backward(ctx: Any, dx0: torch.Tensor) -> torch.Tensor:
        dx = None
        if ctx.needs_input_grad[0]:
            (lohi,) = ctx.saved_tensors
            dx = _idwt3(dx0, lohi)
        return dx, None


class IDWT3D(torch.autograd.Function):
    """Performs the 3d inverse dwt with a custom backward pass that uses less memory

    Args:
        x (torch.Tensor): average and details coefs of shape [N,8,D,C,H,W]
        lohi (torch.Tensor): filter bank of shape [2,K]

    Returns:
        torch.Tensor: average, detail coefs of shape [N,C,D_out,H_out,W_out]
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lohi: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lohi)
        return _idwt3(x, lohi)

    @staticmethod
    def backward(ctx: Any, dx0: torch.Tensor) -> torch.Tensor:
        dx = None
        if ctx.needs_input_grad[0]:
            (lohi,) = ctx.saved_tensors

            dx = _dwt3(dx0, lohi)

        return dx, None
