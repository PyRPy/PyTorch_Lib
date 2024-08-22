
#  21  Time series --------------------------------------------------------

# Forecasting electricity demand
# data
library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)
library(lubridate)

# Tidy Temporal Data Frames and Tools
library(tsibble)
# Feature Extraction and Statistics for Time Series
library(feasts)
# Diverse Datasets for 'tsibble'
library(tsibbledata)

library(torch)
library(luz)

vic_elec

# decompose
decomp <- vic_elec %>%
  filter(year(Date) == 2012) %>%
  model(STL(Demand)) %>%
  components()

decomp %>% autoplot()

# in one month
decomp <- vic_elec %>%
  filter(year(Date) == 2012, month(Date) == 1) %>%
  model(STL(Demand)) %>%
  components()

decomp %>% autoplot()

# predict the next value
# prepare the input data
demand_dataset <- dataset(
  name = "demand_dataset",
  initialize = function(x, n_timesteps, sample_frac = 1) {
    self$n_timesteps <- n_timesteps
    self$x <- torch_tensor((x - train_mean) / train_sd)

    n <- length(self$x) - self$n_timesteps

    self$starts <- sort(sample.int(
      n = n,
      size = n * sample_frac
    ))
  },
  .getitem = function(i) {
    start <- self$starts[i]
    end <- start + self$n_timesteps - 1

    list(
      x = self$x[start:end],
      y = self$x[end + 1]
    )
  },
  .length = function() {
    length(self$starts)
  }
)

# split the data into train, validation and test
demand_hourly <- vic_elec %>%
  index_by(Hour = floor_date(Time, "hour")) %>%
  summarise(
    Demand = sum(Demand))

demand_train <- demand_hourly %>%
  filter(year(Hour) == 2012) %>%
  as_tibble() %>%
  select(Demand) %>%
  as.matrix()

demand_valid <- demand_hourly %>%
  filter(year(Hour) == 2013) %>%
  as_tibble() %>%
  select(Demand) %>%
  as.matrix()

demand_test <- demand_hourly %>%
  filter(year(Hour) == 2014) %>%
  as_tibble() %>%
  select(Demand) %>%
  as.matrix()

train_mean <- mean(demand_train)
train_sd <- sd(demand_train)

n_timesteps <- 7 * 24
train_ds <- demand_dataset(demand_train, n_timesteps)
valid_ds <- demand_dataset(demand_valid, n_timesteps)
test_ds <- demand_dataset(demand_test, n_timesteps)

dim(train_ds[1]$x)
dim(train_ds[1]$y)

batch_size <- 128

train_dl <- train_ds %>%
  dataloader(batch_size = batch_size, shuffle = TRUE)
valid_dl <- valid_ds %>%
  dataloader(batch_size = batch_size)
test_dl <- test_ds %>%
  dataloader(batch_size = length(test_ds))

b <- train_dl %>%
  dataloader_make_iter() %>%
  dataloader_next()

dim(b$x)
dim(b$y)

# model
model <- nn_module(
  initialize = function(input_size,
                        hidden_size,
                        dropout = 0.2,
                        num_layers = 1,
                        rec_dropout = 0) {
    self$num_layers <- num_layers

    self$rnn <- nn_lstm(
      input_size = input_size,
      hidden_size = hidden_size,
      num_layers = num_layers,
      dropout = rec_dropout,
      batch_first = TRUE
    )

    self$dropout <- nn_dropout(dropout)
    self$output <- nn_linear(hidden_size, 1)
  },
  forward = function(x) {
    (x %>%
       # these two are equivalent
       # (1)
       # take output tensor,restrict to last time step
       self$rnn())[[1]][, dim(x)[2], ] %>%
      # (2)
      # from list of state tensors,take the first,
      # and pick the final layer
      # self$rnn())[[2]][[1]][self$num_layers, , ] %>%
      self$dropout() %>%
      self$output()
  }
)

# training
input_size <- 1
hidden_size <- 32
num_layers <- 2
rec_dropout <- 0.2

model <- model %>%
  setup(optimizer = optim_adam, loss = nn_mse_loss()) %>%
  set_hparams(
    input_size = input_size,
    hidden_size = hidden_size,
    num_layers = num_layers,
    rec_dropout = rec_dropout
  )

rates_and_losses <- model %>%
  lr_finder(train_dl, start_lr = 1e-3, end_lr = 1)
rates_and_losses %>% plot()

# fit the model
fitted <- model %>%
  fit(train_dl, epochs = 5, valid_data = valid_dl,
      callbacks = list(
        luz_callback_early_stopping(patience = 3),
        luz_callback_lr_scheduler(
          lr_one_cycle,
          max_lr = 0.1,
          epochs = 50,
          steps_per_epoch = length(train_dl),
          call_on = "on_batch_end")
      ),
      verbose = TRUE)

plot(fitted)

# test / evaluate
evaluate(fitted, test_dl)

# plot and inspect
demand_viz <- demand_hourly %>%
  filter(year(Hour) == 2014, month(Hour) == 12)

demand_viz_matrix <- demand_viz %>%
  as_tibble() %>%
  select(Demand) %>%
  as.matrix()

viz_ds <- demand_dataset(demand_viz_matrix, n_timesteps)
viz_dl <- viz_ds %>% dataloader(batch_size = length(viz_ds))

preds <- predict(fitted, viz_dl)
preds <- preds$to(device = "cpu") %>% as.matrix()
preds <- c(rep(NA, n_timesteps), preds)

pred_ts <- demand_viz %>%
  add_column(forecast = preds * train_sd + train_mean) %>%
  pivot_longer(-Hour) %>%
  update_tsibble(key = name)

pred_ts %>%
  autoplot() +
  scale_colour_manual(values = c("#08c5d1", "#00353f")) +
  theme_minimal() +
  theme(legend.position = "None")


# feeding data, changes
demand_dataset <- dataset(
  name = "demand_dataset",
  initialize = function(x,
                        n_timesteps,
                        n_forecast,
                        sample_frac = 1) {
    self$n_timesteps <- n_timesteps
    self$n_forecast <- n_forecast
    self$x <- torch_tensor((x - train_mean) / train_sd)

    n <- length(self$x) -
      self$n_timesteps - self$n_forecast + 1

    self$starts <- sort(sample.int(
      n = n,
      size = n * sample_frac
    ))
  },
  .getitem = function(i) {
    start <- self$starts[i]
    end <- start + self$n_timesteps - 1

    list(
      x = self$x[start:end],
      y = self$x[(end + 1):(end + self$n_forecast)]$
        squeeze(2)
    )
  },
  .length = function() {
    length(self$starts)
  }
)

n_timesteps <- 7 * 24
n_forecast <- 7 * 24

train_ds <- demand_dataset(
  demand_train,
  n_timesteps,
  n_forecast,
  sample_frac = 1
)
valid_ds <- demand_dataset(
  demand_valid,
  n_timesteps,
  n_forecast,
  sample_frac = 1
)
test_ds <- demand_dataset(
  demand_test,
  n_timesteps,
  n_forecast
)

batch_size <- 128
train_dl <- train_ds %>%
  dataloader(batch_size = batch_size, shuffle = TRUE)
valid_dl <- valid_ds %>%
  dataloader(batch_size = batch_size)
test_dl <- test_ds %>%
  dataloader(batch_size = length(test_ds))

# Forecasting multiple time steps ahead
# model output changes : multi-layer perceptron
model <- nn_module(
  initialize = function(input_size,
                        hidden_size,
                        linear_size,
                        output_size,
                        dropout = 0.2,
                        num_layers = 1,
                        rec_dropout = 0) {
    self$num_layers <- num_layers

    self$rnn <- nn_lstm(
      input_size = input_size,
      hidden_size = hidden_size,
      num_layers = num_layers,
      dropout = rec_dropout,
      batch_first = TRUE
    )

    self$dropout <- nn_dropout(dropout)
    self$mlp <- nn_sequential(
      nn_linear(hidden_size, linear_size),
      nn_relu(),
      nn_dropout(dropout),
      nn_linear(linear_size, output_size)
    )
  },
  forward = function(x) {
    x <- self$rnn(x)[[2]][[1]][self$num_layers, , ] %>%
      self$mlp()
  }
)

# traning
input_size <- 1
hidden_size <- 32
linear_size <- 512
dropout <- 0.5
num_layers <- 2
rec_dropout <- 0.2

model <- model %>%
  setup(optimizer = optim_adam, loss = nn_mse_loss()) %>%
  set_hparams(
    input_size = input_size,
    hidden_size = hidden_size,
    linear_size = linear_size,
    output_size = n_forecast,
    num_layers = num_layers,
    rec_dropout = rec_dropout
  )

rates_and_losses <- model %>% lr_finder(
  train_dl,
  start_lr = 1e-4,
  end_lr = 0.5
)

rates_and_losses %>% plot()

# run the iterations
fitted <- model %>%
  fit(train_dl, epochs = 10, valid_data = valid_dl,
      callbacks = list(
        luz_callback_early_stopping(patience = 3),
        luz_callback_lr_scheduler(
          lr_one_cycle,
          max_lr = 0.01,
          epochs = 100,
          steps_per_epoch = length(train_dl),
          call_on = "on_batch_end")
      ),
      verbose = TRUE)

plot(fitted)

# evaluate
evaluate(fitted, test_dl)
demand_viz <- demand_hourly %>%
  filter(year(Hour) == 2014, month(Hour) == 12)

demand_viz_matrix <- demand_viz %>%
  as_tibble() %>%
  select(Demand) %>%
  as.matrix()

# visualization
n_obs <- nrow(demand_viz_matrix)

viz_ds <- demand_dataset(
  demand_viz_matrix,
  n_timesteps,
  n_forecast
)
viz_dl <- viz_ds %>%
  dataloader(batch_size = length(viz_ds))

preds <- predict(fitted, viz_dl)
preds <- preds$to(device = "cpu") %>%
  as.matrix()

example_preds <- vector(mode = "list", length = 3)
example_indices <- c(1, 201, 401)

for (i in seq_along(example_indices)) {
  cur_obs <- example_indices[i]
  example_preds[[i]] <- c(
    rep(NA, n_timesteps + cur_obs - 1),
    preds[cur_obs, ],
    rep(
      NA,
      n_obs - cur_obs + 1 - n_timesteps - n_forecast
    )
  )
}

pred_ts <- demand_viz %>%
  select(Demand) %>%
  add_column(
    p1 = example_preds[[1]] * train_sd + train_mean,
    p2 = example_preds[[2]] * train_sd + train_mean,
    p3 = example_preds[[3]] * train_sd + train_mean) %>%
  pivot_longer(-Hour) %>%
  update_tsibble(key = name)

pred_ts %>%
  autoplot() +
  scale_colour_manual(
    values = c(
      "#08c5d1", "#00353f", "#ffbf66", "#d46f4d"
    )
  ) +
  theme_minimal() +
  theme(legend.position = "None")
