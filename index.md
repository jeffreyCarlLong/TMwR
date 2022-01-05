---
knit: "bookdown::render_book"
title: "Tidy Modeling with R"
author: ["Max Kuhn and Julia Silge"]
date: "Version 0.0.1.9010 (2022-01-01)"
site: bookdown::bookdown_site
description: "The tidymodels framework is a collection of R packages for modeling and machine learning using tidyverse principles. This book provides a thorough introduction to how to use tidymodels, and an outline of good methodology and statistical practice for phases of the modeling process."
github-repo: tidymodels/TMwR
twitter-handle: topepos
documentclass: book
classoption: 11pt
bibliography: [TMwR.bib]
biblio-style: apalike
link-citations: yes
colorlinks: yes
---

# Hello World {-} 

This is the website for _Tidy Modeling with R_. This book is a guide to using a new collection of software in the R programming language for model building, and it has two main goals: 

- First and foremost, this book provides an introduction to **how to use** our software to create models. We focus on a dialect of R called _[the tidyverse](https://www.tidyverse.org/)_ that is designed to be a better interface for common tasks using R. If you've never heard of or used the tidyverse, Chapter \@ref(tidyverse) provides an introduction. In this book, we demonstrate how the tidyverse can be used to produce high quality models. The tools used to do this are referred to as the _tidymodels packages_. 

- Second, we use the tidymodels packages to **encourage good methodology and statistical practice**. Many models, especially complex predictive or machine learning models, can work very well on the data at hand but may fail when exposed to new data. Often, this issue is due to poor choices made during the development and/or selection of the models. Whenever possible, our software, documentation, and other materials attempt to prevent these and other pitfalls. 

This book is not intended to be a comprehensive reference on modeling techniques; we suggest other resources to learn such nuances. For general background on the most common type of model, the linear model, we suggest @fox08.  For predictive models, @apm is a good resource. Also, @fes is referenced heavily here, mostly because it is freely available online.  For machine learning methods, @Goodfellow is an excellent (but formal) source of information. In some cases, we describe models that are used in this text but in a way that is less mathematical, and hopefully more intuitive. 

:::rmdnote
Investigating and analyzing data are an important part of the model process, and an excellent resource on this topic is @wickham2016.
:::

We do not assume that readers have extensive experience in model building and statistics. Some statistical knowledge is required, such as random sampling, variance, correlation, basic linear regression, and other topics that are usually found in a basic undergraduate statistics or data analysis course. 

:::rmdwarning
_Tidy Modeling with R_ is currently a work in progress. As we create it, this website is updated. Be aware that, until it is finalized, the content and/or structure of the book may change. 
:::

This openness also allows users to contribute if they wish. Most often, this comes in the form of correcting typos, grammar, and other aspects of our work that could use improvement. Instructions for making contributions can be found in the [`contributing.md`](https://github.com/tidymodels/TMwR/blob/main/contributing.md) file. Also, be aware that this effort has a code of conduct, which can be found at [`code_of_conduct.md`](https://github.com/tidymodels/TMwR/blob/main/code_of_conduct.md). 

The tidymodels packages are fairly young in the software lifecycle. We will do our best to maintain backwards compatibility and, at the completion of this work, will archive and tag the specific versions of software that were used to produce it. 

This book was written in [RStudio](http://www.rstudio.com/ide/) using [bookdown](http://bookdown.org/). The [`tmwr.org`](https://tmwr.org) website is hosted via [Netlify](http://netlify.com/), and automatically built after every push by [GitHub Actions](https://help.github.com/actions). The complete source is available on [GitHub](https://github.com/tidymodels/TMwR). We generated all plots in this book using [ggplot2](https://ggplot2.tidyverse.org/) and its black and white theme (`theme_bw()`). This version of the book was built with R version 4.1.2 (2021-11-01), [pandoc](https://pandoc.org/) version 2.14.0.3, and the following packages:


|package         |version    |source                                                               |
|:---------------|:----------|:--------------------------------------------------------------------|
|applicable      |0.0.1.2    |CRAN (R 4.1.0)                                                       |
|av              |0.6.0      |CRAN (R 4.1.0)                                                       |
|baguette        |0.1.1      |CRAN (R 4.1.0)                                                       |
|beans           |0.1.0      |CRAN (R 4.1.0)                                                       |
|bestNormalize   |1.8.2      |CRAN (R 4.1.0)                                                       |
|bookdown        |0.24       |CRAN (R 4.1.0)                                                       |
|broom           |0.7.10     |CRAN (R 4.1.0)                                                       |
|corrplot        |0.92       |CRAN (R 4.1.0)                                                       |
|corrr           |0.4.3      |CRAN (R 4.1.0)                                                       |
|Cubist          |0.3.0      |CRAN (R 4.1.0)                                                       |
|DALEXtra        |2.1.1      |CRAN (R 4.1.0)                                                       |
|dials           |0.0.10     |CRAN (R 4.1.0)                                                       |
|digest          |0.6.29     |CRAN (R 4.1.0)                                                       |
|dimRed          |NA         |NA                                                                   |
|discrim         |0.1.3      |CRAN (R 4.1.0)                                                       |
|doMC            |1.3.7      |CRAN (R 4.1.0)                                                       |
|dplyr           |1.0.7      |CRAN (R 4.1.0)                                                       |
|earth           |5.3.1      |CRAN (R 4.1.0)                                                       |
|embed           |0.1.5      |CRAN (R 4.1.0)                                                       |
|fastICA         |NA         |NA                                                                   |
|finetune        |0.1.0      |CRAN (R 4.1.0)                                                       |
|forcats         |0.5.1      |CRAN (R 4.1.0)                                                       |
|ggforce         |0.3.3      |CRAN (R 4.1.0)                                                       |
|ggplot2         |3.3.5      |CRAN (R 4.1.0)                                                       |
|glmnet          |4.1-3      |CRAN (R 4.1.0)                                                       |
|gridExtra       |2.3        |CRAN (R 4.1.0)                                                       |
|infer           |1.0.0      |CRAN (R 4.1.0)                                                       |
|kableExtra      |1.3.4      |CRAN (R 4.1.0)                                                       |
|kernlab         |0.9-29     |CRAN (R 4.1.0)                                                       |
|kknn            |1.3.1      |CRAN (R 4.1.0)                                                       |
|klaR            |NA         |NA                                                                   |
|knitr           |1.37       |CRAN (R 4.1.0)                                                       |
|learntidymodels |NA         |NA                                                                   |
|lime            |NA         |NA                                                                   |
|lme4            |1.1-27.1   |CRAN (R 4.1.0)                                                       |
|lubridate       |1.8.0      |CRAN (R 4.1.0)                                                       |
|mda             |NA         |NA                                                                   |
|mixOmics        |NA         |NA                                                                   |
|modeldata       |0.1.1      |CRAN (R 4.1.0)                                                       |
|nlme            |3.1-153    |CRAN (R 4.1.2)                                                       |
|nnet            |7.3-16     |CRAN (R 4.1.2)                                                       |
|parsnip         |0.1.7      |CRAN (R 4.1.0)                                                       |
|patchwork       |1.1.1      |CRAN (R 4.1.0)                                                       |
|poissonreg      |0.1.1      |CRAN (R 4.1.0)                                                       |
|prettyunits     |1.1.1      |CRAN (R 4.1.0)                                                       |
|probably        |0.0.6      |CRAN (R 4.1.0)                                                       |
|pscl            |NA         |NA                                                                   |
|purrr           |0.3.4      |CRAN (R 4.1.0)                                                       |
|ranger          |0.13.1     |CRAN (R 4.1.0)                                                       |
|recipes         |0.1.17     |CRAN (R 4.1.0)                                                       |
|rlang           |0.4.12     |CRAN (R 4.1.0)                                                       |
|rmarkdown       |2.11       |CRAN (R 4.1.2)                                                       |
|rpart           |4.1-15     |CRAN (R 4.1.2)                                                       |
|rsample         |0.1.1      |CRAN (R 4.1.0)                                                       |
|rstanarm        |2.21.1     |CRAN (R 4.1.0)                                                       |
|rules           |0.1.2      |CRAN (R 4.1.0)                                                       |
|sessioninfo     |1.2.2      |CRAN (R 4.1.0)                                                       |
|stacks          |0.2.1      |CRAN (R 4.1.0)                                                       |
|stringr         |1.4.0      |CRAN (R 4.1.0)                                                       |
|svglite         |2.0.0      |CRAN (R 4.1.0)                                                       |
|themis          |0.1.4.9000 |Github (tidymodels/themis\@b402b7768156be4feb1236ac9617104075c10995) |
|tibble          |3.1.6      |CRAN (R 4.1.0)                                                       |
|tidymodels      |0.1.4      |CRAN (R 4.1.0)                                                       |
|tidyposterior   |0.1.0      |CRAN (R 4.1.0)                                                       |
|tidyverse       |1.3.1      |CRAN (R 4.1.0)                                                       |
|tune            |0.1.6      |CRAN (R 4.1.0)                                                       |
|uwot            |0.1.11     |CRAN (R 4.1.0)                                                       |
|workflows       |0.2.4      |CRAN (R 4.1.0)                                                       |
|workflowsets    |0.1.0      |CRAN (R 4.1.0)                                                       |
|xgboost         |1.5.0.2    |CRAN (R 4.1.0)                                                       |
|yardstick       |0.0.9      |CRAN (R 4.1.0)                                                       |
