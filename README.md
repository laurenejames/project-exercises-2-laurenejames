Class project: Bayesian linear regresssion
------------------------------------------

The class project is building an R package for Bayesian linear regression. The first few weeks this just involves exercises for working with multivariate normal distributions and model matrices, but later in the class you will be developing the full functionality of an R package.

### Working with model matrices in R

The way we specify both feature vectors and model matrices in R is a *formula*. A Formula is created as an expression containing the tilde symbol, ~, and the target variable should be put to the left and the explanatory variables on the right.

R has quite a rich syntax for specifying formula, and if you are interested you should read the documentation by writing

``` r
?formula
```

in the R shell.

For the linear model, we would write `y ~ x`. The intercept variable is implicitly there; you don't need to tell R that you want the feature vector to include the "-1", instead, you would have to remove it explicitly. You can also specify polynomial feature vectors, but R interprets multiplication, `*`, as something involving interaction between variables.[1] To specify that you want the second order polynomial of *x*, you need to write `y ~ I(x^2) + x`. The function `I` is the identity function and using it here makes R interpret the `x^2` as squaring the number `x` instead of trying to interpret it as part of the formula specification. If you *only* want to fit the square of *x*, you would just write `y ~ I(x^2)`. For a general *n* degree polynomial you can use `y ~ poly(x,n, raw=TRUE)`.

To fit our linear model we need data for two things. In the model we have already implemented we had vectors **x** and **y**, but in the general case the prediction variable **x** should be replaced with the model matrix **Φ**. From **Φ** and **y** we can fit the model.

R has functions for getting both from a formula and data. It isn't *quite* straightforward, though, because of scoping rules. If you write a formula somewhere in your code, you want the variables in the formula to refer to the variables in the scope where you are. Not somewhere else where the code *might* look at the formula. So the formula needs to capture the current scope -- similar to how a closure captures the scope around it. On the other hand, you also want to be able to provide data directly to models via data frames. Quite often, the data you want to fit is found as columns in a data frame, not as individual variables in scope. Sometimes it is even a mix.

The function `model.frame` lets you capture what you need for collecting data relevant for a formula. It will know about the scope of the formula, but you can add data through a data frame as well. Think of it as a `data.frame`, just with a bit more information about the data that it gets from analysing the formula.

We can see all of this in action in the small example below:

``` r
predictors <- data.frame(x = rnorm(5), z = rnorm(5))
y <- with(predictors, rnorm(5, mean = 3*x + 5*z + 2))

model <- y ~ x + z

model.frame(model, data = predictors)
```

    ##             y           x          z
    ## 1 -0.09482537 -0.01379934 -0.4963711
    ## 2  0.48560533  1.40208224 -0.9177324
    ## 3  5.11721257  1.82654717 -0.4119407
    ## 4  3.58517854 -0.47799938  0.3897072
    ## 5  2.51797080  1.26527234 -0.5932914

Here we have two predictor variables, `x` and `z`, in a data frame, and we simulated the response variable, `y`, in the global scope. The model we create using the formula `y ~ x + z` \#ifdef SIMPLE\_MATH (which means *ϕ*(*x*, *z*)′ = (1, *x*, *z*)) \#else (which means *ϕ*(*x*, *z*)<sup>*T*</sup> = (1, *x*, *z*)) \#endif and we construct a model frame from this that contains the data for all the variables used in the formula.

The way the model frame gets created, R first looks in the data frame it gets for a variable, and if it is there it uses that data, if it is not, it uses the data it can find in the scope of the formula. If it cannot find it at all, it will, of course, report an error.

The data frame is also used to construct expressions from variables. In the scope you might have the variable *x* but not the variable *x*<sup>2</sup> where the latter is needed for constructing a model matrix. The `model.frame` function will construct it for you.

``` r
x <- runif(10)
model.frame(~ x + I(x^2))
```

    ##             x       I(x^2)
    ## 1  0.49588311 0.245900....
    ## 2  0.40752510 0.166076....
    ## 3  0.72069213 0.519397....
    ## 4  0.21732463 0.047229....
    ## 5  0.47222876 0.223000....
    ## 6  0.62126435 0.385969....
    ## 7  0.47914926 0.229584....
    ## 8  0.09773652 0.009552....
    ## 9  0.14342595 0.020571....
    ## 10 0.11456454 0.013125....

In this example we don't have a response variable for the formula; you don't necessarily need one. You need it to be able to extract the vector **y** of course, so we do need one for our linear model fitting, but R doesn't necessarily need one.

Once you have a model frame, you can get the model matrix using the function `model.matrix`. It needs to know the formula and the model frame (the former to know the feature function *ϕ* and the latter to know the data we are fitting).

Below we build two models, one where we fit a line that goes through *y* = 0 and the second where we allow the line to intersect the *y* axis at an arbitrary point.

Notice how the data frames are the same -- the variables used in both models are the same -- but the model matrices differ.

``` r
x <- runif(10)
y <- rnorm(10, mean=x)

model.no.intercept <- y ~ x + 0
(frame.no.intercept <- model.frame(model.no.intercept))
```

    ##             y          x
    ## 1  -1.2053275 0.09066605
    ## 2   0.4832548 0.92369474
    ## 3   1.0229064 0.37509789
    ## 4   1.2642583 0.87297836
    ## 5   1.6212914 0.56643592
    ## 6   0.3232016 0.42263896
    ## 7   1.8485863 0.85474625
    ## 8   0.4537268 0.70962193
    ## 9  -1.2221129 0.33802869
    ## 10  0.7131609 0.22554979

``` r
model.matrix(model.no.intercept, frame.no.intercept)
```

    ##             x
    ## 1  0.09066605
    ## 2  0.92369474
    ## 3  0.37509789
    ## 4  0.87297836
    ## 5  0.56643592
    ## 6  0.42263896
    ## 7  0.85474625
    ## 8  0.70962193
    ## 9  0.33802869
    ## 10 0.22554979
    ## attr(,"assign")
    ## [1] 1

``` r
model.with.intercept <- y ~ x
(frame.with.intercept <- model.frame(model.with.intercept))
```

    ##             y          x
    ## 1  -1.2053275 0.09066605
    ## 2   0.4832548 0.92369474
    ## 3   1.0229064 0.37509789
    ## 4   1.2642583 0.87297836
    ## 5   1.6212914 0.56643592
    ## 6   0.3232016 0.42263896
    ## 7   1.8485863 0.85474625
    ## 8   0.4537268 0.70962193
    ## 9  -1.2221129 0.33802869
    ## 10  0.7131609 0.22554979

``` r
model.matrix(model.with.intercept, frame.with.intercept)
```

    ##    (Intercept)          x
    ## 1            1 0.09066605
    ## 2            1 0.92369474
    ## 3            1 0.37509789
    ## 4            1 0.87297836
    ## 5            1 0.56643592
    ## 6            1 0.42263896
    ## 7            1 0.85474625
    ## 8            1 0.70962193
    ## 9            1 0.33802869
    ## 10           1 0.22554979
    ## attr(,"assign")
    ## [1] 0 1

The target vector, or response variable, **y**, can be extracted from the data frame as well. You don't need the formula this time because the data frame actually remembers which variable is the response variable. You can get it from the model frame using the function `model.response`:

``` r
model.response(frame.with.intercept)
```

    ##          1          2          3          4          5          6 
    ## -1.2053275  0.4832548  1.0229064  1.2642583  1.6212914  0.3232016 
    ##          7          8          9         10 
    ##  1.8485863  0.4537268 -1.2221129  0.7131609

### Exercise

#### Building model matrices

Build a function that takes as input a formula and optionally, through the `...` variable, a data frame and build the model matrix from the formula and optional data.

``` r
model_matrix <- function(formula, ...) {
  model.matrix(formula, ...)
}

```

#### Fitting general models

Extend the function you wrote earlier for fitting lines to a function that can fit any formula.

``` r
# YOUR SOLUTION HERE
```

### Model matrices without response variables

Building model matrices this way is all good and well when you have all the variables needed for the model frame, but what happens when you don't have the target value? You need the target value to fit the parameters of your model, of course, but later on, you want to predict targets for new data points where you do *not* know the target, so how do you build the model matrix then?

With some obviously fake data the situation could look like this:

``` r
training.data <- data.frame(x = runif(5), y = runif(5))
frame <- model.frame(y ~ x, training.data)
model.matrix(y ~ x, frame)
```

    ##   (Intercept)         x
    ## 1           1 0.0869882
    ## 2           1 0.9564345
    ## 3           1 0.6475028
    ## 4           1 0.9244734
    ## 5           1 0.7893293
    ## attr(,"assign")
    ## [1] 0 1

``` r
predict.data <- data.frame(x = runif(5))
```

where of course we get a problem when trying to build the frame without knowing the target variable `y`.

``` r
frame <- model.frame(y ~ x, predict.data)
```

    ## Error in model.frame.default(y ~ x, predict.data): variable lengths differ (found for 'x')

If only there were a way to remove the response variable from the formula! And there is.

The function `delete.response` does just that. You cannot call it directly on a formula. R first needs to collect some information for this function to work, unlike the other functions we saw above. But you can combine it with the function `terms` to get a formula without the response variable that you can then use to build a model matrix for data where you don't know the target values.

``` r
responseless.formula <- delete.response(terms(y ~ x))
frame <- model.frame(responseless.formula, predict.data)
model.matrix(responseless.formula, frame)
```

    ##   (Intercept)         x
    ## 1           1 0.9941321
    ## 2           1 0.6588978
    ## 3           1 0.7669882
    ## 4           1 0.2221738
    ## 5           1 0.9089075
    ## attr(,"assign")
    ## [1] 0 1

### Exercise

#### Model matrices for new data

Write a function that takes as input a formula and a data frame as input that does *not* contain the response variable and build the model matrix for that.

``` r
model_matrix_responseless <- function(responseless.formula, ...) {
  model.matrix(responseless.formula, ...)
}
```

#### Predicting new targets

Update the function you wrote last week for predicting the values for new variables to work on models fitted to general formula. If it doesn't already permit this, you should also extend it so it can take more than one such data point. Make the input for new data points come in the form of a data frame.

``` r
# YOUR SOLUTION HERE
```

[1] In formulas `x*z` means `x + z + x:z` where `x:z` is the interaction between `x` and `z` -- in practice the product of their numbers -- so `y ~ x*z` means *ϕ*(*x*, *z*)=(1, *x*, *z*, *x* ⋅ *z*)).
