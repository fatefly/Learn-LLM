import argparse
from typing import List
import os

import torch
import triton
import triton.language as tl

@triton.jit
def demo1(x_ptr):
    range = tl.arange(0, 8)
    print(range)
    x = tl.load(x_ptr + range, range < 5, 0)
    print(x)

def run_demo1():
    print("Demo1 Output: ")
    demo1[(1,1,1)](torch.ones(4,3))

@triton.jit
def demo2(x_ptr):
    i_range = tl.arange(0, 8)[:, None]
    j_range = tl.arange(0, 4)[None, :]
    range = i_range*4 + j_range
    print(range)
    x = tl.load(x_ptr + range, (i_range < 4)&(j_range < 3), 0)
    print(x)

def run_demo2():
    print("Demo2 Output: ")
    demo2[(1, 1, 1)](torch.ones(4, 4))

@triton.jit
def demo3(z_ptr):
    range = tl.arange(0, 8)
    z = tl.store(z_ptr + range, 10, range < 5)

def run_demo3():
    print("Demo3 Output: ")
    z = torch.ones(4, 3)
    demo3[(1, 1, 1)](z)
    print(z)

@triton.jit
def demo4(x_ptr):
    pid = tl.program_id(0)
    range = tl.arange(0, 8) + pid * 8
    x = tl.load(x_ptr + range, range < 20)
    print("Print for each", pid, x)

def run_demo4():
    print("Demo4 Output: ")
    x = torch.ones(2, 4, 4)
    demo4[(3, 1, 1)](x)

@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    off_x = tl.arange(0, B0)
    x = tl.load(x_ptr + off_x)
    x = x + 10.0
    tl.store(z_ptr + off_x, x)

@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    block_id = tl.program_id(0)
    off_x = tl.arange(0, B0) + block_id * B0
    mask = off_x < N0
    x = tl.load(x_ptr + off_x, mask=mask)
    x = x + 10.0
    tl.load(z_ptr + off_x, mask=mask)
    return

@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    off_x = tl.arange(0, B0)
    off_y = tl.arange(0, B1)
    off_z = off_y * B0 + off_x[None, :]
    x = tl.load(x_ptr + off_x)
    y = tl.load(y_ptr + off_y)
    z = x[None, :] + y[:, None]
    tl.store(z_ptr + off_z, z)
    return

@triton.jit
def add_vec_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    block_id_x = tl.program_id(0)
    block_id_y = tl.program_id(1)
    off_x = block_id_x * B0 + tl.arange(0, B0)
    off_y = block_id_y * B1 + tl.arange(0, B1)
    off_z = off_y[:, None] * N0 + off_x[None, :]
    mask_x = off_x < N0
    mask_y = off_y < N1
    mask_z = mask_y[:, None] & mask_x[None, :]
    x = tl.load(x_ptr + off_x, mask=mask_x)
    y = tl.load(y_ptr + off_y, mask=mask_y)
    z = y[:, None] + x[None, :]
    tl.store(z_ptr + off_z, z, mask=mask_z)
    return

@triton.jit
def mul_relu_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    block_id_x = tl.program_id(0)
    block_id_y = tl.program_id(1)
    off_x = block_id_x * B0 + tl.arange(0, B0)
    off_y = block_id_y * B1 + tl.arange(0, B1)
    off_z = off_y[:, None] * N0 + off_x[None, :]
    mask_x = off_x < N0
    mask_y = off_y < N1
    mask_z = mask_y[:, None] & mask_x[None, :]
    x = tl.load(x_ptr + off_x, mask=mask_x)
    y = tl.load(y_ptr + off_y, mask=mask_y)
    z = x[None, :]*y[:, None]
    relu_z = tl.where(z > 0, z, 0.0)
    tl.store(z_ptr + off_z, relu_z, mask=mask_z)
    return

@triton.jit
def mul_relu_block_back_kernel(x_ptr, y_ptr, dz_ptr, dx_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    block_id_i = tl.program_id(0)
    block_id_j = tl.program_id(1)
    off_i = block_id_i * B0 + tl.arange(0, B0)
    off_j = block_id_j * B1 + tl.arange(0, B1)
    off_ji = off_j[:, None] * N0 + off_i[None, :]
    mask_i = off_i < N0
    mask_j = off_j < N1
    mask_ji = mask_j[:, None] & mask_i[None, :]

    x = tl.load(x_ptr + off_ji, mask=mask_ji)
    y = tl.load(y_ptr + off_j, mask=mask_j)
    dz = tl.load(dz_ptr + off_ji, mask=mask_ji)
    df = tl.where(x * y[:, None] > 0, 1.0, 0.0)
    dxy_x = y[:, None]
    dx = df * dxy_x * dz
    tl.store(dx_ptr + off_ji, dx, mask=mask_ji)
    return

@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    block_id_i = tl.program_id(0)
    off_i = block_id_i * B0 + tl.arange(0, B0)
    mask_i = off_i < N0
    z = tl.zeros([B0], dtype=tl.float32)

    for id_j in tl.range(0, T, B1):
        off_j = id_j + tl.arange(0, B1)
        off_ij = off_i[:, None]*T + off_j[None, :]
        mask_j = off_j < T
        mask_ij = mask_i[:, None] & mask_j[None, :]
        x = tl.load(x_ptr + off_ij, mask=mask_ij)
        z += tl.sum(x, axis=1)
    tl.store(z_ptr + off_i, mask=mask_i)
    return

