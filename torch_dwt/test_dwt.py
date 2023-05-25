import pytest
import torch
import pywt
import numpy as np
from .functional import dwt, idwt, dwt2, idwt2, dwt3, idwt3

torch.manual_seed(0)


@pytest.mark.parametrize("n, wavelet", [(1, "haar"), (127, "db2"), (128, "db4")])
def test_dwt(n: int, wavelet: str):
    wavelet = pywt.Wavelet(wavelet)
    x = torch.rand(8, 7, n)  # NxCxW
    target = torch.from_numpy(
        np.array(pywt.dwt(x.numpy(), wavelet, mode="zero", axis=-1))
    ).permute(1, 0, 2, 3)
    result = dwt(x, wavelet)
    torch.testing.assert_close(result, target)


@pytest.mark.parametrize(
    "n, wavelet",
    [
        (2, "haar"),
        (128, "db4"),
    ],
)
def test_idwt(n: int, wavelet: str):
    wavelet = pywt.Wavelet(wavelet)
    x = torch.rand(8, 7, n)  # NxCxW
    dwt_coefs = dwt(x, wavelet)
    result = idwt(dwt_coefs, wavelet)
    torch.testing.assert_close(result, x)


@pytest.mark.parametrize(
    "shape, wavelet", [((1, 1), "haar"), ((127, 32), "db2"), ((128, 128), "db4")]
)
def test_dwt2(shape: tuple[int, int], wavelet: str):
    wavelet = pywt.Wavelet(wavelet)
    x = torch.rand(8, 7, *shape)  # NxCxHxW

    target = pywt.dwt2(x.numpy(), wavelet, mode="zero")
    # we reorder coefs to cA,cV,CH,cD because this is the pytorch conv output
    target_coefs = torch.from_numpy(np.array([target[0], *target[1]]))[
        [0, 2, 1, 3], ...
    ].permute(1, 0, 2, 3, 4)

    result = dwt2(x, wavelet)
    torch.testing.assert_close(result, target_coefs)


@pytest.mark.parametrize(
    "shape, wavelet", [((2, 2), "haar"), ((256, 32), "db2"), ((128, 128), "db4")]
)
def test_idwt2(shape: tuple[int, int], wavelet: str):
    wavelet = pywt.Wavelet(wavelet)
    x = torch.rand(8, 7, *shape)  # NxCxHxW
    dwt_coefs = dwt2(x, wavelet)
    result = idwt2(dwt_coefs, wavelet)
    torch.testing.assert_close(result, x)


@pytest.mark.parametrize(
    "shape, wavelet",
    [((1, 1, 1), "haar"), ((64, 64, 64), "db2"), ((64, 32, 127), "db4")],
)
def test_dwt3(shape: tuple[int, int, int], wavelet: str):
    wavelet = pywt.Wavelet(wavelet)
    x = torch.rand(8, 7, *shape)  # NxCxDxHxW

    target = pywt.dwtn(x.numpy(), wavelet, mode="zero", axes=[-3, -2, -1])
    keys = list(target.keys())
    target_coefs = torch.from_numpy(np.array([target[k] for k in keys])).permute(
        1, 0, 2, 3, 4, 5
    )

    result = dwt3(x, wavelet)
    torch.testing.assert_close(result, target_coefs)


@pytest.mark.parametrize(
    "shape, wavelet",
    [((2, 2, 2), "haar"), ((64, 64, 64), "db2"), ((64, 32, 128), "db4")],
)
def test_idwt3(shape: tuple[int, int, int], wavelet: str):
    wavelet = pywt.Wavelet(wavelet)
    x = torch.rand(8, 7, *shape)  # NxCxHxW
    dwt_coefs = dwt3(x, wavelet)
    result = idwt3(dwt_coefs, wavelet)
    torch.testing.assert_close(result, x)
