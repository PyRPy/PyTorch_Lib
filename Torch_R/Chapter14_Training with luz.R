
# Training with luz -------------------------------------------------------
# https://skeydan.github.io/Deep-Learning-and-Scientific-Computing-with-R-torch/training_with_luz.html
# a linear regression example
# data
library(torch)
library(luz)

# input dimensionality (number of input features)
d_in <- 3
# number of observations in training set
n <- 1000

x <- torch_randn(n, d_in)
coefs <- c(0.2, -1.3, -0.5)
y <- x$matmul(coefs)$unsqueeze(2) + torch_randn(n, 1)

ds <- tensor_dataset(x, y)

dl <- dataloader(ds, batch_size = 100, shuffle = TRUE)

# model
# dimensionality of hidden layer
d_hidden <- 32
# output dimensionality (number of predicted features)
d_out <- 1

net <- nn_module(
  initialize = function(d_in, d_hidden, d_out) {
    self$net <- nn_sequential(
      nn_linear(d_in, d_hidden),
      nn_relu(),
      nn_linear(d_hidden, d_out)
    )
  },
  forward = function(x) {
    self$net(x)
  }
)

# training
fitted <- net %>%
  setup(loss = nn_mse_loss(), optimizer = optim_adam) %>%
  set_hparams(
    d_in = d_in,
    d_hidden = d_hidden, d_out = d_out
  ) %>%
  fit(dl, epochs = 200)

# Integrating training, validation, and test
train_ids <- sample(1:length(ds), size = 0.6 * length(ds))
valid_ids <- sample(
  setdiff(1:length(ds), train_ids),
  size = 0.2 * length(ds)
)
test_ids <- setdiff(
  1:length(ds),
  union(train_ids, valid_ids)
)

train_ds <- dataset_subset(ds, indices = train_ids)
valid_ds <- dataset_subset(ds, indices = valid_ids)
test_ds <- dataset_subset(ds, indices = test_ids)

train_dl <- dataloader(train_ds,
                       batch_size = 100, shuffle = TRUE
)
valid_dl <- dataloader(valid_ds, batch_size = 100)
test_dl <- dataloader(test_ds, batch_size = 100)

# training with validation and test
fitted <- net %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_mae())
  ) %>%
  set_hparams(
    d_in = d_in,
    d_hidden = d_hidden, d_out = d_out
  ) %>%
  fit(train_dl, epochs = 200, valid_data = valid_dl)

# prediction
fitted %>% predict(test_dl)

# evaluate performance on the test set
fitted %>% evaluate(test_dl)

# callbacks (weights will be saved)
fitted <- net %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_mae())
  ) %>%
  set_hparams(d_in = d_in,
              d_hidden = d_hidden,
              d_out = d_out) %>%
  fit(
    train_dl,
    epochs = 200,
    valid_data = valid_dl,
    callbacks = list(
      luz_callback_model_checkpoint(path = "./models/",
                                    save_best_only = TRUE),
      luz_callback_early_stopping(patience = 10)
    )
  )

# Early stopping at epoch 64 of 200
