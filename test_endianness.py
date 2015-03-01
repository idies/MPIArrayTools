#! /usr/bin/env python3
import numpy as np


data = np.memmap(
    'data_native',
    mode = 'r',
    dtype = np.float32)
print(data)

data = np.memmap(
    'data_internal',
    mode = 'r',
    dtype = np.float32)
print(data)

data = np.memmap(
    'data_external32',
    mode = 'r',
    dtype = '>f4')
print(data)
