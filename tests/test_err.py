import torch

a = torch.zeros(10, dtype=torch.complex128)
b = torch.ones(10, dtype=torch.complex64)
try:
    a.index_put_((torch.arange(10),), b, accumulate=True)
except Exception as e:
    print("A is complex128, B is complex64:", e)

a = torch.zeros(10, dtype=torch.complex64)
b = torch.ones(10, dtype=torch.complex128)
try:
    a.index_put_((torch.arange(10),), b, accumulate=True)
except Exception as e:
    print("A is complex64, B is complex128:", e)
