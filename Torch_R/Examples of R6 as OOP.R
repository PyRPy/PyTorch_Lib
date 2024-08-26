
# R6 as OOP ---------------------------------------------------------------
# the code comes from : https://adv-r.hadley.nz/r6.html
library(R6)

###class and method
# use list as container to house class name, variables / attributes and
# functions/ methods

Accumulator <- R6Class("Accumulator", list(
  sum = 0,
  add = function(x = 1) {
    self$sum <- self$sum + x
    invisible(self)
  })
)

Accumulator

x <- Accumulator$new()
x$add(4)
x$sum
x$sum() # Error: attempt to apply non-function

### method chaining (like in python)
x$add(10)$add(10)$sum


### $initialize() in R6 like contructor in Java or Python
Person <- R6Class("Person", list(
  name = NULL,
  age = NA,
  initialize = function(name, age = NA) {
    stopifnot(is.character(name), length(name) == 1)
    stopifnot(is.numeric(age), length(age) == 1)

    self$name <- name
    self$age <- age
  }
))

hadley <- Person$new("Hadley", age = "thirty-eight")
#> Error in initialize(...): is.numeric(age) is not TRUE

hadley <- Person$new("Hadley", age = 38)

### print()
Person <- R6Class("Person", list(
  name = NULL,
  age = NA,
  initialize = function(name, age = NA) {
    self$name <- name
    self$age <- age
  },
  print = function(...) {
    cat("Person: \n")
    cat("  Name: ", self$name, "\n", sep = "")
    cat("  Age:  ", self$age, "\n", sep = "")
    invisible(self)
  }
))

hadley2 <- Person$new("Hadley")
hadley2

### Inheritance
AccumulatorChatty <- R6Class("AccumulatorChatty",
                             inherit = Accumulator,
                             public = list(
                               add = function(x = 1) {
                                 cat("Adding ", x, "\n", sep = "")
                                 super$add(x = x)
                               }
                             )
)

x2 <- AccumulatorChatty$new()
x2$add(10)$add(1)$sum

### Introspection
class(hadley2)

### Privacy
Person <- R6Class("Person",
                  public = list(
                    initialize = function(name, age = NA) {
                      private$name <- name
                      private$age <- age
                    },
                    print = function(...) {
                      cat("Person: \n")
                      cat("  Name: ", private$name, "\n", sep = "")
                      cat("  Age:  ", private$age, "\n", sep = "")
                    }
                  ),
                  private = list(
                    age = NA,
                    name = NULL
                  )
)

hadley3 <- Person$new("Hadley")
hadley3
hadley3$name # NULL, cannot access

### Active fields
Rando <- R6::R6Class("Rando", active = list(
  random = function(value) {
    if (missing(value)) {
      runif(1)
    } else {
      stop("Can't set `$random`", call. = FALSE)
    }
  }
))
x <- Rando$new()
x$random # like python's generator

### Reference semantics
y1 <- Accumulator$new()
y2 <- y1

y1$add(10)
c(y1 = y1$sum, y2 = y2$sum) # same, use clone()

### R6 fields, be aware of
TemporaryFile <- R6Class("TemporaryFile", list(
  path = NULL,
  initialize = function() {
    self$path <- tempfile()
  },
  finalize = function() {
    message("Cleaning up ", self$path)
    unlink(self$path)
  }
))

library(DBI)
TemporaryDatabase <- R6Class("TemporaryDatabase", list(
  con = NULL,
  file = TemporaryFile$new(),
  initialize = function() {
    self$con <- DBI::dbConnect(RSQLite::SQLite(), path = file$path)
  },
  finalize = function() {
    DBI::dbDisconnect(self$con)
  }
))

db_a <- TemporaryDatabase$new()
db_b <- TemporaryDatabase$new()

db_a$file$path == db_b$file$path # TRUE



