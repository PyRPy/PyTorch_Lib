
#  20  Tabular data -------------------------------------------------------

library(torch)
library(luz)

library(purrr)
library(readr)
library(dplyr)

# uci <- "https://archive.ics.uci.edu"
# ds_path <- "ml/machine-learning-databases/heart-disease"
# ds_file <- "processed.cleveland.data"
#
# download.file(
#   file.path(uci, ds_path, ds_file),
#   destfile = "resources/tabular-heart.csv"
# )

heart_df <- read_csv(
  "processed.cleveland.csv",
  col_names = c(
    "age",
    # 1 = male; 0 = female
    "sex",
    # chest pain type
    # (1 = typical angina, 2 = atypical angina,
    #  3 = non-anginal pain, 4 = asymptomatic)
    "pain_type",
    # in mm Hg on admission
    "resting_blood_pressure",
    # serum cholesterol in mg/dl
    "chol",
    # > 120 mg/dl, true (1) or false (0)
    "fasting_blood_sugar",
    # 0 = normal, 1 = ST-T wave abnormality
    # (T wave inversions and/or ST elevation
    # or depression of > 0.05 mV),
    # 2 = probable or definite left ventricular
    # hypertrophy by Estes' criteria
    "rest_ecg",
    # during exercise
    "max_heart_rate",
    # exercise induced angina (1 = yes, 0 = no),
    "ex_induced_angina",
    # ST depression induced by exercise relative to rest
    "old_peak",
    # slope of the peak exercise ST segment
    # (1 = upsloping, 2 = flat, 3 = downsloping)
    "slope",
    # number of major vessels (0-3) colored by fluoroscopy
    "ca",
    # 3 = normal; 6 = fixed defect; 7 = reversible defect
    "thal",
    # 1-4 = yes; 0 = no
    "heart_disease"
  ),
  na = "?")

head(heart_df)

# check dataset
which(is.na(heart_df), arr.ind = TRUE)
heart_df %>% group_by(thal) %>% summarise(n())
heart_df %>% group_by(ca) %>% summarise(n())

heart_dataset <- dataset(
  initialize = function(df) {
    self$x_cat <- self$get_categorical(df)
    self$x_num <- self$get_numerical(df)
    self$y <- self$get_target(df)
  },
  .getitem = function(i) {
    x_cat <- self$x_cat[i, ]
    x_num <- self$x_num[i, ]
    y <- self$y[i]
    list(x = list(x_cat, x_num), y = y)
  },
  .length = function() {
    dim(self$y)[1]
  },
  get_target = function(df) {
    heart_disease <- ifelse(df$heart_disease > 0, 1, 0)
    heart_disease
  },
  get_numerical = function(df) {
    df %>%
      select(
        -(c(
          heart_disease, pain_type,
          rest_ecg, slope, ca, thal
        ))
      ) %>%
      mutate(across(.fns = scale)) %>%
      as.matrix()
  },
  get_categorical = function(df) {
    df$ca <- ifelse(is.na(df$ca), 999, df$ca)
    df$thal <- ifelse(is.na(df$thal), 999, df$thal)
    df %>%
      select(
        pain_type, rest_ecg, slope, ca, thal
      ) %>%
      mutate(
        across(.fns = compose(as.integer, as.factor))
      ) %>%
      as.matrix()
  }
)

ds <- heart_dataset(heart_df)
ds[1]

# prepare the data inputs
train_indices <- sample(
  1:nrow(heart_df), size = floor(0.8 * nrow(heart_df)))
valid_indices <- setdiff(
  1:nrow(heart_df), train_indices)

train_ds <- dataset_subset(ds, train_indices)
train_dl <- train_ds %>%
  dataloader(batch_size = 256, shuffle = TRUE)

valid_ds <- dataset_subset(ds, valid_indices)
valid_dl <- valid_ds %>%
  dataloader(batch_size = 256, shuffle = FALSE)

# imbedding
embedding_module <- nn_module(
  initialize = function(cardinalities, embedding_dim) {
    self$embeddings <- nn_module_list(
      lapply(
        cardinalities,
        function(x) {
          nn_embedding(
            num_embeddings = x, embedding_dim = embedding_dim
          )
        }
      )
    )
  },
  forward = function(x) {
    embedded <- vector(
      mode = "list",
      length = length(self$embeddings)
    )
    for (i in 1:length(self$embeddings)) {
      embedded[[i]] <- self$embeddings[[i]](x[, i])
    }
    torch_cat(embedded, dim = 2)
  }
)

# model construction
model <- nn_module(
  initialize = function(cardinalities,
                        num_numerical,
                        embedding_dim,
                        fc1_dim,
                        fc2_dim) {
    self$embedder <- embedding_module(
      cardinalities,
      embedding_dim
    )
    self$fc1 <- nn_linear(
      embedding_dim * length(cardinalities) + num_numerical,
      fc1_dim
    )
    self$drop1 <- nn_dropout(p = 0.7)
    self$fc2 <- nn_linear(fc1_dim, fc2_dim)
    self$drop2 <- nn_dropout(p = 0.7)
    self$output <- nn_linear(fc2_dim, 1)
  },
  forward = function(x) {
    embedded <- self$embedder(x[[1]])
    all <- torch_cat(list(embedded, x[[2]]), dim = 2)
    score <- all %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$drop1() %>%
      self$fc2() %>%
      nnf_relu() %>%
      self$drop2() %>%
      self$output()
    score[, 1]
  }
)

# cardinalities of categorical features
cardinalities <- heart_df %>%
  select(pain_type, rest_ecg, slope, ca, thal) %>%
  mutate(across(.fns = as.factor)) %>%
  summarise(across(.fns = nlevels))

# cardinalities of categorical features,
# adjusted for presence of NAs in ca and thal
cardinalities <- cardinalities + c(0, 0, 0, 1, 1)

# number of numerical features
num_numerical <- ncol(heart_df) - length(cardinalities) - 1

embedding_dim <- 7

fc1_dim <- 32
fc2_dim <- 32

# training
fitted <- model %>%
  setup(
    optimizer = optim_adam,
    loss = nn_bce_with_logits_loss(),
    metrics = luz_metric_binary_accuracy_with_logits()
  ) %>%
  set_hparams(
    cardinalities = cardinalities,
    num_numerical = num_numerical,
    embedding_dim = embedding_dim,
    fc1_dim = fc1_dim, fc2_dim
  ) %>%
  set_opt_hparams(lr = 0.001) %>%
  fit(train_dl,
      epochs = 200,
      valid_data = valid_dl,
      callbacks = list(
        luz_callback_early_stopping(patience = 10)
      ),
      verbose = TRUE
  )
