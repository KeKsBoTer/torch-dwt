# 3D Discrete Wavelet Transform (DWT) im Pytorch

This package implements the 1D,2D,3D [Discrete Wavelet Transform](https://en.wikipedia.org/wiki/Discrete_wavelet_transform) and inverse DWT (IDWT) in Pytorch.
The package was heavily inspired by [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets) and extends its functionality into the third dimension.

The wavelets are provided by the [PyWavelets](https://github.com/PyWavelets/pywt) package.

All operations in this package are fully differentiable. 

## Installation

```
git clone https://github.com/KeKsBoTer/torch-dwt
cd torch-dwt
pip install -e .
```


## Example Usage

## 3D

```python
from torch_dwt.functional import dwt3,idwt3
import torch

# 8 images with 3 color channels and size of 100x100
x = torch.rand(8,3,100,100,100)
coefs = dwt3(x,"sym2") # coefs of shape (1,2,3,50)
# reconstruct signal from coefficients
y = idwt3(coefs,"sym2")
```


## 2D

```python
from torch_dwt.functional import dwt2,idwt2
import torch

# 8 images with 3 color channels and size of 100x100
x = torch.rand(8,3,100,100)
coefs = dwt2(x,"db2") # coefs of shape (1,2,3,50)
# reconstruct signal from coefficients
y = idwt2(coefs,"db2")
```

## 1D

```python
from torch_dwt.functional import dwt,idwt
import torch

# batch of size 8 with 3 channels
x = torch.rand(8,3,100)
coefs = dwt(x,"haar") # coefs of shape (1,2,3,50)
# reconstruct signal from coefficients
y = idwt(coefs,"haar")
```

## Testing

For testing we compare our implementation againts [PyWavelets](https://github.com/PyWavelets/pywt).
This command runs the tests:
```bash
# navigate into torch-dwt directory
pytest .
```