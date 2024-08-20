
# Modularizing the neural network -----------------------------------------

# simulated data
# input dimensionality (number of input features)
d_in <- 3
# number of observations in training set
n <- 100

x <- torch_randn(n, d_in)
coefs <- c(0.2, -1.3, -0.5)
y <- x$matmul(coefs)$unsqueeze(2) + torch_randn(n, 1)

# network for a simple neural net
# dimensionality of hidden layer
d_hidden <- 32
# output dimensionality (number of predicted features)
d_out <- 1

net <- nn_sequential(
  nn_linear(d_in, d_hidden),
  nn_relu(),
  nn_linear(d_hidden, d_out)
)

# train the network
opt <- optim_adam(net$parameters)

### training loop --------------------------------------

for (t in 1:200) {

  ### -------- Forward pass --------
  y_pred <- net(x)

  ### -------- Compute loss --------
  loss <- nnf_mse_loss(y_pred, y)
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")

  ### -------- Backpropagation --------
  opt$zero_grad()
  loss$backward()

  ### -------- Update weights --------
  opt$step()

}
print(y_pred, 10)
print(y, 10)
print((y_pred - y), 10) # check difference

