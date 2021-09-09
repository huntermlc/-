from numpy.core.fromnumeric import shape
from numpy.core.numeric import ones
import torch
import numpy as np
from torch.functional import tensordot
data=[[1,2],[3,4]]
x_data=torch.tensor(data)

np_array=np.array(data)
x_np=torch.from_numpy(np_array)

x_one=torch.ones_like(x_data)
print(f"Ones Tenso：\n{x_one}\n")

x_rand=torch.rand_like(x_data,dtype=torch.float)
print(f"Random Tensor:\n{x_rand}\n")

shape=(2,3,)
rand_tensor=torch.rand(shape)
ones_tensor=torch.ones(shape)
zeros_tensor=torch.zeros(shape)
print(f"Random Tensor:\n{rand_tensor}\n")
print(f"Ones Tensor:\n{ones_tensor}\n")
print(f"Zeros Tensor:\n{zeros_tensor}\n")

tensor=torch.rand(3,4)
print(f"Shape of tensor:{tensor.shape}")
print(f"Dtype of tensor:{tensor.dtype}")
print(f"Device tenspr os stored on:{tensor.device}")


tensor1=torch.ones(5,5)
print('First row:',tensor1[0])
print('First column:',tensor1[:,0])
print('Last column:',tensor1[...,-1])
tensor1[:,2]=0
print(tensor1)

t1=torch.cat([tensor1,tensor1,tensor1,tensor1],dim=-2)
print(t1)

