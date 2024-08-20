
# Chapter 15 Image classification -----------------------------------------

library(torch)
library(torchvision)
library(luz)

set.seed(777)
torch_manual_seed(777)

# load data
dir <- "~/.torch-datasets"

train_ds <- tiny_imagenet_dataset(
  dir,
  download = TRUE,
  transform = function(x) {
    x %>%
      transform_to_tensor()
  }
)

valid_ds <- tiny_imagenet_dataset(
  dir,
  split = "val",
  transform = function(x) {
    x %>%
      transform_to_tensor()
  }
)

train_dl <- dataloader(train_ds,
                       batch_size = 128,
                       shuffle = TRUE
)

valid_dl <- dataloader(valid_ds, batch_size = 128)

batch <- train_dl %>%
  dataloader_make_iter() %>%
  dataloader_next()

dim(batch$x)
batch$y

# network
convnet <- nn_module(
  "convnet",
  initialize = function() {
    self$features <- nn_sequential(
      nn_conv2d(3, 64, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_conv2d(64, 128, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_conv2d(128, 256, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_conv2d(256, 512, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_max_pool2d(kernel_size = 2),
      nn_conv2d(512, 1024, kernel_size = 3, padding = 1),
      nn_relu(),
      nn_adaptive_avg_pool2d(c(1, 1))
    )
    self$classifier <- nn_sequential(
      nn_linear(1024, 1024),
      nn_relu(),
      nn_linear(1024, 1024),
      nn_relu(),
      nn_linear(1024, 200)
    )
  },
  forward = function(x) {
    x <- self$features(x)$squeeze()
    x <- self$classifier(x)
    x
  }
)

# train the model
fitted <- convnet %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    )
  ) %>%
  fit(train_dl,
      epochs = 5,
      valid_data = valid_dl,
      verbose = TRUE
  )

# too slow to run on a laptop without a decent GPU
