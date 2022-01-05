


# Screening many models  {#workflow-sets}

We introduced workflow sets in Chapter \@ref(workflows) and demonstrated how to use them with resampled data sets in Chapter \@ref(compare). In this chapter, we discuss these sets of multiple modeling workflows in more detail and describe a use case where they can be helpful. 

For projects with new data sets that have not yet been well understood, a data practitioner may need to screen many combinations of models and preprocessors. It is common to have little or no _a priori_ knowledge about which method will work best with a novel data set. 

:::rmdnote
A good strategy is to spend some initial effort trying a variety of modeling approaches, determine what works best, then invest additional time tweaking/optimizing a small set of models.   
:::

Workflow sets provide a user interface to create and manage this process. We'll also demonstrate how to evaluate these models efficiently using the racing methods discussed in Section \@ref(racing-example).

## Modeling concrete strength

Let's use the concrete data from _Applied Predictive Modeling_ [@apm] as an example. Chapter 10 of that book demonstrated models to predict the compressive strength of concrete mixtures using the ingredients as predictors. A wide variety of models were evaluated with different predictor sets and preprocessing needs. How can workflow sets make the large scale testing of models easier? 

First, let's define the data splitting and resampling schemes.


```r
library(tidymodels)
tidymodels_prefer()
data(concrete, package = "modeldata")
glimpse(concrete)
#> Rows: 1,030
#> Columns: 9
#> $ cement               <dbl> 540.0, 540.0, 332.5, 332.5, 198.6, 266.0, 380.0, 380.…
#> $ blast_furnace_slag   <dbl> 0.0, 0.0, 142.5, 142.5, 132.4, 114.0, 95.0, 95.0, 114…
#> $ fly_ash              <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
#> $ water                <dbl> 162, 162, 228, 228, 192, 228, 228, 228, 228, 228, 192…
#> $ superplasticizer     <dbl> 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0…
#> $ coarse_aggregate     <dbl> 1040.0, 1055.0, 932.0, 932.0, 978.4, 932.0, 932.0, 93…
#> $ fine_aggregate       <dbl> 676.0, 676.0, 594.0, 594.0, 825.5, 670.0, 594.0, 594.…
#> $ age                  <int> 28, 28, 270, 365, 360, 90, 365, 28, 28, 28, 90, 28, 2…
#> $ compressive_strength <dbl> 79.99, 61.89, 40.27, 41.05, 44.30, 47.03, 43.70, 36.4…
```

The `compressive_strength` column is the outcome. The `age` predictor tells us the age of the concrete sample at testing in days (concrete strengthens over time) and the rest of the predictors like `cement` and `water` are concrete components in units of kilograms per cubic meter.

:::rmdwarning
There are some cases in this data set where the same concrete formula was tested multiple times. We'd rather not include these replicate mixtures as individual data points since they might be distributed across both the training and test set. Doing so might artificially inflate our performance estimates.  
:::

To address this, we will use the mean compressive strength per concrete mixture for modeling. 


```r
concrete <- 
   concrete %>% 
   group_by(across(-compressive_strength)) %>% 
   summarize(compressive_strength = mean(compressive_strength),
             .groups = "drop")
nrow(concrete)
#> [1] 992
```

Let's split the data using the default 3:1 ratio of training-to-test and resample the training set using five repeats of 10-fold cross-validation. 


```r
set.seed(1501)
concrete_split <- initial_split(concrete, strata = compressive_strength)
concrete_train <- training(concrete_split)
concrete_test  <- testing(concrete_split)

set.seed(1502)
concrete_folds <- 
   vfold_cv(concrete_train, strata = compressive_strength, repeats = 5)
```

Some models (notably neural networks, K-nearest neighbors, and support vector machines) require predictors that have been centered and scaled, so some model workflows will require recipes with these preprocessing steps. For other models, a traditional response surface design model expansion (i.e., quadratic and two-way interactions) is a good idea. For these purposes, we create two recipes: 


```r
normalized_rec <- 
   recipe(compressive_strength ~ ., data = concrete_train) %>% 
   step_normalize(all_predictors()) 

poly_recipe <- 
   normalized_rec %>% 
   step_poly(all_predictors()) %>% 
   step_interact(~ all_predictors():all_predictors())
```

For the models, we use the the <span class="pkg">parsnip</span> addin to create a set of model specifications: 


```r
library(rules)
library(baguette)

linear_reg_spec <- 
   linear_reg(penalty = tune(), mixture = tune()) %>% 
   set_engine("glmnet")

nnet_spec <- 
   mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>% 
   set_engine("nnet", MaxNWts = 2600) %>% 
   set_mode("regression")

mars_spec <- 
   mars(prod_degree = tune()) %>%  #<- use GCV to choose terms
   set_engine("earth") %>% 
   set_mode("regression")

svm_r_spec <- 
   svm_rbf(cost = tune(), rbf_sigma = tune()) %>% 
   set_engine("kernlab") %>% 
   set_mode("regression")

svm_p_spec <- 
   svm_poly(cost = tune(), degree = tune()) %>% 
   set_engine("kernlab") %>% 
   set_mode("regression")

knn_spec <- 
   nearest_neighbor(neighbors = tune(), dist_power = tune(), weight_func = tune()) %>% 
   set_engine("kknn") %>% 
   set_mode("regression")

cart_spec <- 
   decision_tree(cost_complexity = tune(), min_n = tune()) %>% 
   set_engine("rpart") %>% 
   set_mode("regression")

bag_cart_spec <- 
   bag_tree() %>% 
   set_engine("rpart", times = 50L) %>% 
   set_mode("regression")

rf_spec <- 
   rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
   set_engine("ranger") %>% 
   set_mode("regression")

xgb_spec <- 
   boost_tree(tree_depth = tune(), learn_rate = tune(), loss_reduction = tune(), 
              min_n = tune(), sample_size = tune(), trees = tune()) %>% 
   set_engine("xgboost") %>% 
   set_mode("regression")

cubist_spec <- 
   cubist_rules(committees = tune(), neighbors = tune()) %>% 
   set_engine("Cubist") 
```

The analysis in @apm specifies that the neural network should have up to 27 hidden units in the layer. The `parameters()` function creates a parameter set that we modify to have the correct parameter range.


```r
nnet_param <- 
   nnet_spec %>% 
   parameters() %>% 
   update(hidden_units = hidden_units(c(1, 27)))
```

How can we match these models to their recipes, tune them, then evaluate their performance efficiently? A workflow set offers a solution. 

## Creating the workflow set

Workflow sets take named lists of preprocessors and model specifications and combine them into an object containing multiple workflows. There are three possible kinds of preprocessors: 

* A standard R formula
* A recipe object (prior to estimation/prepping)
* A <span class="pkg">dplyr</span>-style selector to choose the outcome and predictors

As a first workflow set example, let's combine the recipe that only standardizes the predictors to the nonlinear models that require that the predictors be in the same units. 


```r
normalized <- 
   workflow_set(
      preproc = list(normalized = normalized_rec), 
      models = list(SVM_radial = svm_r_spec, SVM_poly = svm_p_spec, 
                    KNN = knn_spec, neural_network = nnet_spec)
   )
normalized
#> # A workflow set/tibble: 4 × 4
#>   wflow_id                  info             option    result    
#>   <chr>                     <list>           <list>    <list>    
#> 1 normalized_SVM_radial     <tibble [1 × 4]> <opts[0]> <list [0]>
#> 2 normalized_SVM_poly       <tibble [1 × 4]> <opts[0]> <list [0]>
#> 3 normalized_KNN            <tibble [1 × 4]> <opts[0]> <list [0]>
#> 4 normalized_neural_network <tibble [1 × 4]> <opts[0]> <list [0]>
```

Since there is only a single preprocessor, this function creates a set of workflows with this value. If the preprocessor contained more than one entry, the function would create all combinations of preprocessors and models. 

The `wflow_id` column is automatically created but can be modified using a call to `mutate()`. The `info` column contains a tibble with some identifiers and the workflow object. The workflow can be extracted: 


```r
normalized %>% extract_workflow(id = "normalized_KNN")
#> ══ Workflow ═════════════════════════════════════════════════════════════════════════
#> Preprocessor: Recipe
#> Model: nearest_neighbor()
#> 
#> ── Preprocessor ─────────────────────────────────────────────────────────────────────
#> 1 Recipe Step
#> 
#> • step_normalize()
#> 
#> ── Model ────────────────────────────────────────────────────────────────────────────
#> K-Nearest Neighbor Model Specification (regression)
#> 
#> Main Arguments:
#>   neighbors = tune()
#>   weight_func = tune()
#>   dist_power = tune()
#> 
#> Computational engine: kknn
```

The `option` column is a placeholder for any arguments to use when we evaluate the workflow. For example, to add the neural network parameter object:  


```r
normalized <- 
   normalized %>% 
   option_add(param_info = nnet_param, id = "normalized_neural_network")
normalized
#> # A workflow set/tibble: 4 × 4
#>   wflow_id                  info             option    result    
#>   <chr>                     <list>           <list>    <list>    
#> 1 normalized_SVM_radial     <tibble [1 × 4]> <opts[0]> <list [0]>
#> 2 normalized_SVM_poly       <tibble [1 × 4]> <opts[0]> <list [0]>
#> 3 normalized_KNN            <tibble [1 × 4]> <opts[0]> <list [0]>
#> 4 normalized_neural_network <tibble [1 × 4]> <opts[1]> <list [0]>
```

When a function from the <span class="pkg">tune</span> or <span class="pkg">finetune</span> package is used to tune (or resample) the workflow, this argument will be used. 

The `result` column is a placeholder for the output of the tuning or resampling functions.  

For the other nonlinear models, let's create another workflow set that uses <span class="pkg">dplyr</span> selectors for the outcome and predictors: 


```r
model_vars <- 
   workflow_variables(outcomes = compressive_strength, 
                      predictors = everything())

no_pre_proc <- 
   workflow_set(
      preproc = list(simple = model_vars), 
      models = list(MARS = mars_spec, CART = cart_spec, CART_bagged = bag_cart_spec,
                    RF = rf_spec, boosting = xgb_spec, Cubist = cubist_spec)
   )
no_pre_proc
#> # A workflow set/tibble: 6 × 4
#>   wflow_id           info             option    result    
#>   <chr>              <list>           <list>    <list>    
#> 1 simple_MARS        <tibble [1 × 4]> <opts[0]> <list [0]>
#> 2 simple_CART        <tibble [1 × 4]> <opts[0]> <list [0]>
#> 3 simple_CART_bagged <tibble [1 × 4]> <opts[0]> <list [0]>
#> 4 simple_RF          <tibble [1 × 4]> <opts[0]> <list [0]>
#> 5 simple_boosting    <tibble [1 × 4]> <opts[0]> <list [0]>
#> 6 simple_Cubist      <tibble [1 × 4]> <opts[0]> <list [0]>
```

Finally, the set that uses nonlinear terms and interactions with the appropriate models are assembled: 


```r
with_features <- 
   workflow_set(
      preproc = list(full_quad = poly_recipe), 
      models = list(linear_reg = linear_reg_spec, KNN = knn_spec)
   )
```

These objects are tibbles with the extra class of `workflow_set`. Row binding does not affect the state of the sets and the result is itself a workflow set:


```r
all_workflows <- 
   bind_rows(no_pre_proc, normalized, with_features) %>% 
   # Make the workflow ID's a little more simple: 
   mutate(wflow_id = gsub("(simple_)|(normalized_)", "", wflow_id))
all_workflows
#> # A workflow set/tibble: 12 × 4
#>   wflow_id    info             option    result    
#>   <chr>       <list>           <list>    <list>    
#> 1 MARS        <tibble [1 × 4]> <opts[0]> <list [0]>
#> 2 CART        <tibble [1 × 4]> <opts[0]> <list [0]>
#> 3 CART_bagged <tibble [1 × 4]> <opts[0]> <list [0]>
#> 4 RF          <tibble [1 × 4]> <opts[0]> <list [0]>
#> 5 boosting    <tibble [1 × 4]> <opts[0]> <list [0]>
#> 6 Cubist      <tibble [1 × 4]> <opts[0]> <list [0]>
#> # … with 6 more rows
```

## Tuning and evaluating the models

Almost all of these workflows contain tuning parameters. In order to evaluate their performance, we can use the standard tuning or resampling functions (e.g., `tune_grid()` and so on). The `workflow_map()` function will apply the same function to all of the workflows in the set; the default is `tune_grid()`. 

For this example, grid search is applied to each workflow using up to 25 different parameter candidates. There are a set of common options to use with each execution of `tune_grid()`. For example, we will use the same resampling and control objects for each workflow, along with a grid size of 25. The `workflow_map()` function has an additional argument called `seed` that is used to ensure that each execution of `tune_grid()` consumes the same random numbers. 


```r
grid_ctrl <-
   control_grid(
      save_pred = TRUE,
      parallel_over = "everything",
      save_workflow = TRUE
   )

grid_results <-
   all_workflows %>%
   workflow_map(
      seed = 1503,
      resamples = concrete_folds,
      grid = 25,
      control = grid_ctrl
   )
```

The results show that the `option` and `result` columns have been updated:


























