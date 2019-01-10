"""
main code to play around with PyTorch
"""
from __future__ import print_function

import os
import sys
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
os.chdir("../../")
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
import import_folders

# Public Libraries
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import copy as copy
import pandas as pd

# Own graphical library
from graph_lib import gl
# Data Structures Data
import CTimeData as CTD
# Specific utilities
import utilities_lib as ul

plt.close("all") # Close all previous Windows

"""
PyTorch Related library !!
"""

import torch

"""
Basic tensor creation
"""
x = torch.empty(5, 3)
x = torch.rand(5, 3)
x = torch.zeros(5, 3, dtype=torch.long)

## Construct Tensor from data
x = torch.tensor([5.5, 3])


#or create a tensor based on an existing tensor. 
#These methods will reuse properties of the input tensor, e.g. dtype,
# unless new values are provided by user

x = torch.randn_like(x, dtype=torch.float) 

print(x)

## Size is a tuple
x = torch.rand(5, 3)
print(x.size()); print (type(x.size()))



"""
#################### OPERATIONS ########################
Operations
There are multiple syntaxes for operations.
 In the following example, we will take a look at the addition operation.
"""

"""
 Any operation that mutates a tensor in-place is post-fixed with an _.
 For example: 
     x.copy_(y), x.t_(), will change x.
     
"""
x = torch.rand(5, 3)
y = torch.rand(5, 3)
# Addition: syntax 1
print(x + y)
# Addition: syntax 2
print(torch.add(x, y))

# Addition: providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
# Addition in place
y.add_(x)
print(y)


"""
INDEXING !!
You can use standard NumPy-like indexing with all bells and whistles!


"""

print(x[:, 1])
"""
Resizing: If you want to resize/reshape tensor, you can use torch.view:
"""

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())


"""
Conversion from Tensor to Python.
The conversion does not create new data so if you modify the tensor, 
it will modify numpy
"""
## One dimensional value
x = torch.randn(1)
print(x)
print(x.item())

## Convert to numpy:

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

b = b + 3 # If I modify b then they are not the same anymore ?
print(a)
print(b)
a.add_(5)
print(a)
print(b)


"""
CUDA Tensors
Tensors can be moved onto any device using the .to method.

"""

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!





