
# Chapter 6 A neural network from scratch ---------------------------------

library(torch)
x <- torch_randn(100, 3)
x$size()

# addition and multiplication for linear equations
w <- torch_randn(3, 1, requires_grad = TRUE)
b <- torch_zeros(1, 1, requires_grad = TRUE)
y <- x$matmul(w) + b
print(y, n = 10)
print(w)
print(b)
print(y)

# concept of layers

w1 <- torch_randn(3, 8, requires_grad = TRUE)
print(w1)
b1 <- torch_zeros(1, 8, requires_grad = TRUE)
print(b1)

w2 <- torch_randn(8, 1, requires_grad = TRUE)
b2 <- torch_randn(1, 1, requires_grad = TRUE)

# concept of activation functions
t <- torch_tensor(c(-2, 1, 5, -7)) # all negative numbers set to zero
t$relu()

t1 <- torch_tensor(c(1, 2, 3))
t2 <- torch_tensor(c(1, -2, 3))
t1$add(t2)$relu()

t1_clamped <- t1$relu()
t2_clamped <- t2$relu()
t1_clamped$add(t2_clamped) # order makes difference

# concept of loss function
y <- torch_randn(5)
y_pred <- y + 0.01

loss <- (y_pred - y)$pow(2)$mean()

loss

# how to do it in torch
library(torch)

# input dimensionality (number of input features)
d_in <- 3
# number of observations in training set
n <- 100

x <- torch_randn(n, d_in)
coefs <- c(0.2, -1.3, -0.5)
y <- x$matmul(coefs)$unsqueeze(2) + torch_randn(n, 1)

# network
# dimensionality of hidden layer
d_hidden <- 32
# output dimensionality (number of predicted features)
d_out <- 1

# weights connecting input to hidden layer
w1 <- torch_randn(d_in, d_hidden, requires_grad = TRUE)
# weights connecting hidden to output layer
w2 <- torch_randn(d_hidden, d_out, requires_grad = TRUE)

# hidden layer bias
b1 <- torch_zeros(1, d_hidden, requires_grad = TRUE)
# output layer bias
b2 <- torch_zeros(1, d_out, requires_grad = TRUE)

# train the network
learning_rate <- 1e-4

### training loop ----------------------------------------

for (t in 1:200) {

  ### -------- Forward pass --------

  y_pred <- x$mm(w1)$add(b1)$relu()$mm(w2)$add(b2)

  ### -------- Compute loss --------
  loss <- (y_pred - y)$pow(2)$mean()
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")

  ### -------- Backpropagation --------

  # compute gradient of loss w.r.t. all tensors with
  # requires_grad = TRUE
  loss$backward()

  ### -------- Update weights --------

  # Wrap in with_no_grad() because this is a part we don't
  # want to record for automatic gradient computation
  with_no_grad({
    w1 <- w1$sub_(learning_rate * w1$grad)
    w2 <- w2$sub_(learning_rate * w2$grad)
    b1 <- b1$sub_(learning_rate * b1$grad)
    b2 <- b2$sub_(learning_rate * b2$grad)

    # Zero gradients after every pass, as they'd
    # accumulate otherwise
    w1$grad$zero_()
    w2$grad$zero_()
    b1$grad$zero_()
    b2$grad$zero_()
  })

}
print(w2)
print(b2)
print(w1)
