
# Chapter 3 Tensors -------------------------------------------------------

library(torch)
# tensors
t1 <- torch_tensor(1)
t1

t3 <- t1$view(c(1, 1))
t3$shape

c(1)
matrix(1)

# create tensors
torch_tensor(matrix(1:9, ncol = 3))

# tensor and array slice differently
torch_tensor(array(1:24, dim = c(4, 3, 2)))
array(1:24, dim = c(4, 3, 2))
