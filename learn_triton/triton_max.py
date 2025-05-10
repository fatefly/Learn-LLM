import torch
import triton
import triton.language as tl

class VALUES_INDICES:
    def __init__(self, values, indices, func_type):
        self.values = values
        self.indices = indices
        self.func_type = func_type

    def __str__(self):
        return f"func_type={self.func_type}\nvalues={self.values}\nindices={self.indices})"
    
    def __repr__(self):
        return f"func_type={self.func_type}\nvalues={self.values}\nindices={self.indices})"
    
@triton.jit
def max_short(INPUT, VALUES, INDICES, stride0, stride1, M, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pass

@triton.jit
def max_long(INPUT, VALUES, INDICES, stride0, stride1, M, N: tl.constexpr, BLOCK_N: tl.constexpr):
    pass

def triton_max(tensor, axis=-1, keepdim=False):
    tensor = torch.movedim(tensor, axis, -1)
    print(tensor)
    tensor_shape = tensor.shape
    print(tensor_shape)
    tensor = tensor.reshape(-1, tensor_shape[-1])
    print(tensor)
    print(tensor.stride())
    B, D = tensor.shape
    values = torch.empty(B, device=tensor.device, dtype=tensor.dtype)
    indices = torch.empty(B, device=tensor.device, dtype=tensor.dtype)
    if D < 256:
        tmp = triton.next_power_of_2(B)
        BLOCK_M = min(256, tmp)
        BLOCK_N = triton.next_power_of_2(D)
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_M']), )
        max_short[grid](tensor, values, indices, tensor.stride(0), tensor.stride(1), B, D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    else:
        BLOCK_N = min(triton.next_power_of_2(D), 2048)
        max_long[(B,)](tensor, values, indices, tensor.stride(0), tensor.stride(1), B, D, BLOCK_N=BLOCK_N)
    values = values.reshape(*tensor_shape[:-1])
    indices = indices.reshape(*tensor_shape[:-1])
    if keepdim:
        values.unsqueeze_(axis)
    return VALUES_INDICES(values=values, indices=indices, func_type="triton_max")