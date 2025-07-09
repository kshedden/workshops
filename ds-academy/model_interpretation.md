# Model interpretation

## Predictivity

- When considering the relationship between explanatory variables $X$ and an
  outcome or response $Y$, we can focus on the conditional mean (regression
  function) $E[Y|X]$ and the conditional variance ${\rm Var}[Y|X]$. The
  conditional variance reflects the scatter of the data around the conditional
  mean.

- Informally, we say that $Y$ explains $X$ through the conditional mean
  $E[Y|X]$.

- The conditional variance function ${\rm Var}[Y|X]$ is related to the notion
  of
  [heteroscedasticity](https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity).
  If the conditional variance function is not constant, we have different
  levels of scatter in different parts of the predictor space. If the
  conditional variance function is constant, the level of scatter is constant
  throughout the predictor space and we have _homoscedasticity_.

- We can also discuss the mean of the conditional variance $E {\rm var}[Y|X]$
  and the variance of the conditional mean ${\rm var} E[Y|X]$. The former is
  the average level of scatter around the mean, and the latter is a measure of
  the departure of the conditional mean from being a constant function (i.e. a
  measure of heteroscedasticity).

- A model for a quantitative outcome $Y$ can be considered _predictive_ if it
  explains a (non-zero) portion of the variation of $Y$, which can be taken to
  mean that $E[Y|X=x]$ is not a constant function of $x$, or that
  ${\rm var}E[Y|X] > 0$.

- If we consider the
  [law of total variation](https://en.wikipedia.org/wiki/Law_of_total_variance),
  the variance of $Y$ can be decomposed as
  ${\rm Var}[Y] = {\rm Var} E[Y|X] + E {\rm Var}[Y|X]$, where $X$ is a vector
  of predictor variables. Since both ${\rm Var} E[Y|X]$ and $E {\rm Var}[Y|X]$
  must be non-negative, we can interpret ${\rm Var} E[Y|X]$ as the variation
  of $Y$ that is _explained_ by the predictors $X$.

- For a model to be useful, arguably it needs to be predictive to some extent.
  That is, in almost all circumstances, a useful model should satisfy
  ${\rm Var} E[Y|X] > 0$.

  - As an exception to this principle, in some cases it may be sufficient for
    a model to explain something about the variance in $Y$, even when there is
    no variation in $E[Y|X]$. A model is predictive of heteroscedasticity, but
    explains nothing about the mean itself, if ${\rm Var} E[Y|X] = 0$, but
    ${\rm Var} {\rm Var}[Y|X=x]$ is not zero (or, ${\rm Var}[Y|X=x]$ is not
    constant in $x$).

    - Even more esoteric would be predictivity that exclusively informs us
      about other properties of the distribution such as the
      [skewness](https://en.wikipedia.org/wiki/Skewness).

### Measures of predictivity

- The amount of variation of $Y$ that is explained by $X$ can be quantified
  using the
  [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination),
  also known as the $R^2$, which can be defined as
  ${\rm Var}E[Y|X]/{\rm Var}[Y]$. The $R^2$ ranges from $0$ to $1$.
  Specifically, this is the _population_ $R^2$, which captures the total
  information in $X$ about $Y$.

- If we have a prediction $\hat{Y}$ of $Y$, we can consider estimate the $R^2$
  using the _variance ratio_ ${\rm Var}[\hat{Y}]/{\rm Var}[Y]$. This is a
  measure of how well the model used to produce $\hat{Y}$ captures the true
  value $Y$.

  - When $\hat{Y}$ is obtained using
    [ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares),
    ${\rm corr}(\hat{Y}, Y)^2$ is identical to the variance ratio
    ${\rm Var}[\hat{Y}]/{\rm Var}[Y]$.

  - Below we will consider some important practical issues associated with
    estimating the $R^2$, or other measures of predictivity,
    [unbiasedly](<https://en.wikipedia.org/wiki/Bias_(statistics)>).

- The value of $R^2$ that can be considered "useful" is highly
  context-specific. In many applications involving human behavior, or
  considering individual factors within large systems (e.g. contributions of
  genes to phenotypes), $R^2$ values as small as 1% or less can still be
  useful and meaningful.

- Besides the $R^2$, other measures of predictivity are useful in specific
  settings, especially when the variable $Y$ being modeled is not
  quantitative.

- When working with a binary response $Y$ (i.e. $Y$ must equal $0$ or $1$),
  and its predicted value $\hat{Y}$, it is useful to consider the following
  four conditional probabilities:

  - $P(\hat{Y}=1 | Y=1)$ is the
    [sensitivity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity),
    also known as the _true positive rate_ or the _recall_.

  - $P(\hat{Y}=0 | Y=0)$ is the
    [specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity),
    also known as the _true negative rate_.

  - $P(Y=1 | \hat{Y}=1)$ is the
    [positive predictive value](https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values),
    also known as the _precision_.

  - $P(Y=0 | \hat{Y}=0)$ is the
    [negative predictive value](https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values).

- The terminology used here (sensitivity, etc.) is meant to reflect the
  situation where there are different costs associated with different
  predictive errors. We often focus on two types of predictive errors: a
  _false positive_ occurs when $\hat{Y}=1$ but $Y=0$, and a _false negative_
  occurs when $\hat{Y} = 0$ but $Y=1$.

- Many models for a binary outcome $Y$ produce a quantitative score $Z$ rather
  than a binary decision. In this setting, a decision can be made by setting a
  threshold $t$, and setting $\hat{Y}=1$ when $Z>t$ and $\hat{Y}=0$ when
  $Z\le t$. In this setting, higher values of $t$ result in increased
  specificity and decreased sensitivity.

- The
  [receiver operating characteristics curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic),
  or _ROC curve_ is a plot of sensitivity, plotted on the vertical axis,
  against the complement of the specificity (1 - specificity), plotted on the
  horizontal axis.

- The ROC curve is non-decreasing, and in general should lie above the
  diagonal $y=x$. If the ROC curve falls mostly or entirely below the line
  $y=x$, the model is predicting worse than guessing, or worse than always
  predicting the more common outcome. The greater the value of the ROC curve,
  i.e. the higher the sensitivity at a given value of 1 minus specificity, the
  better the performance of the model. High values of the ROC curve toward the
  left side of the plot reflect better performance in very conservative (high
  specificity) regimes, and high values toward the right side of the plot
  reflect better performance in very permissive (low specificity) regimes.

- The area under the ROC curve is known as the AUC or the AUROC and is an
  important metric for the predictive performance of models for binary
  outcomes. The AUC ranges from 0 to 1, with higher values corresponding to
  better predictive performance.

- Insight into the AUC can be gained through its relationship to the
  [Mann-Whitney U statistic](https://en.wikipedia.org/wiki/Mann-Whitney_U_test).
  In this way, we can see that the AUC is equal to
  $P(Z_1 > Z_2 | Y_1=1, Y_2=0)$, where the probability is computed over all
  independent pairs such that $Y_1=1$ and $Y_2=0$. As above, the $Z_j$ are
  quantitative scores reflecting the evidence that $Y=1$. For simplicity, here
  we consider a setting where the $Z$'s are never tied.

- Based on the Mann-Whitney interpretation of the AUC, we see that the AUC
  reflects the concordance between the predictions and the actual outcomes.

- Another measure of predictivity that can be used with binary $Y$ is the
  [Brier score](https://en.wikipedia.org/wiki/Brier_score). This can only be
  used when we have a _probabilistic scoring rule_, which effectively means
  that our quantitative evidence value $Z$ can be interpreted as a
  probability, specifically $Z \approx P(Y=1|X)$. In this case, the Brier
  score can be interpreted as ${\rm Avg} (Y_i - Z_i)^2$.

### Predictivity and first-principles analysis

- The fact that a model is predictive has little to do with whether the model
  captures important mechanistic features of the population of interest. In
  some settings, a mechanistic model can predict poorly while non-mechanistic
  models predicts well.

- Models are based on data that can be measured. Since not all relevant
  variables can be measured, and those that can be measured are impacted by
  measurement error, it follows that all models are in some sense incomplete
  or wrong. This is the basis for the famous statement of George Box that some
  models are useful but
  [all models are wrong](https://en.wikipedia.org/wiki/All_models_are_wrong).

- We can take a useful model to be one that is either predictive and/or
  mechanistically informative. If our only goal is to make predictions, it
  generally does not matter how closely the model captures mechanisms that
  truly cause the variation in $Y$. If our goal is to gain fundamental
  insights into the causes of the variation of $Y$, then it becomes very
  important for the model to reflect the underlying mechanisms, and this is
  the basis of
  [causal inference](https://en.wikipedia.org/wiki/Causal_inference) in
  statistics.

- Predictivity is contingent on the distribution of our predictors $X$ over
  their domain. Put in more formal terms, if the goal is to capture a
  regression function $E[Y|X=x]$ through a model $f(x)$, the performance of
  our model depends both on the extent to which $f(x) \approx E[Y|X=x]$, and
  on the marginal distribution $P_X(x)$ of the predictors. In particular, the
  model will perform better (on average) if the values $x$ where $f(x)$ is a
  poor approximation to $E[Y|X]$ have low probability.

- A more general notion of predictivity considers the possibility of a
  _distribution shift_ between the "training" and "testing" populations. We
  can define $P^{\rm train}_X$ to be the training distribution and
  $P^{\rm test}_X$ to be the testing distribution, so that a distribution
  shift occurs when $P^{\rm train}_X \ne P^{\rm test}_X$.

- A distribution shift is one type of
  [non-stationarity](https://en.wikipedia.org/wiki/Stationarity_process),
  which refers to any situation where the structure of the population we are
  observing drifts over time.

- The presence of non-stationarity usually degrades the _generalization_
  performance of models, and may also impact their basis for making causal
  inferences.

- An extreme form of non-stationarity forces us to _extrapolate_, which occurs
  when the support of $P^{\rm test}_X$ is different from the support of
  $P^{\rm train}_X$. This means that there are regions with positive
  probability under $P^{\rm test}_X$ that have zero probability of occurring
  under $P^{\rm train}_X$.

- Distribution shift refers to the situation where the marginal distribution
  of predictors $P_X$ changes over time. Another form of non-stationarity
  occurs when the probability distribution $P(Y|X)$ or more narrowly the
  conditional mean $E[Y|X]$ changes over time. This form of non-stationarity
  is even more challenging than what is encountered under distribution shift,
  since we now cannot know whether relationships among variables of interest
  are stable over time.

## Estimating predictivity

- The population measures of predictivity all have "plug-in" estimators, but
  these estimators will be biased in the direction of higher predictivity. For
  example, the population $R^2$ is ${\rm cor}(Y, E[Y|X])^2$, and the plug-in
  estimator replaces $E[Y|X]$ with $\hat{Y}$, yielding the estimator
  ${\rm cor}(Y, \hat{Y})^2$. This estimator, like most plug-in measures of
  predictivity, is biased upward due to
  [overfitting](https://en.wikipedia.org/wiki/Overfitting).

- When fitting models with ordinary least squares, the bias in $R^2$ due to
  overfitting can be approximated analytically. If $\hat{R}^2$ is the sample
  $R^2$ and $R^2$ is the population value, then an unbiased estimator of
  $1 - R^2$ is $(1 - \hat{R}^2)n/(n-p)$, where $n$ is the sample size and $p$
  is the number of explanatory variables. Since this adjustment inflates
  $1 - \hat{R}^2$ toward one, the $R^2$ itself is shrunk toward zero. If we
  think of $1 - R^2$ as the unexplained variance, this result shows that the
  unexplained variance is too small when using the plug-in estimate, and this
  can be compensated by inflating it by a factor of $n/(n-p)$.

  - The analysis above is the basis for the _adjusted_ $R^2$.

- For other measures of predictivity (e.g. AUC) and other ways of fitting
  models (besides least squares), there is no general way to analytically
  adjust the sample statistic to make it unbiased. Instead, computational
  techniques are used to debias the statistic.

- The most common general-purpose technique for unbiasedly estimating the
  level of predictivity of a model is
  [cross validation](<https://en.wikipedia.org/wiki/Cross-validation_(statistics)>).

- In k-fold cross validation (CV), the data are partitioned into $k$
  non-overlapping subsets (or "folds"). The model is trained on $k-1$ of the
  subsets, meaning that the model parameters are learned exclusively using
  this portion of the data (which excludes exactly one of the $k$ subsets).
  Let $\hat{\theta}[j]$ denote the model parameters trained when excluding the
  $j^{\rm th}$ subset of data. We can now produce _prediction residuals_
  $y_i - f_{\hat{\theta}[j]}(x_i)$ for each index $i$ in fold $j$. These
  prediction residuals are unaffected by overfitting, since the data used to
  fit the model does not include the observation being predicted.

- Once we have a complete set of prediction residuals (obtained by holding out
  each fold in turn), we can use them to estimate predictivity metrics such as
  the $R^2$, AUC, and Brier score. Due to the use of cross validation, these
  estimates are not upwardly biased due to overfitting. They are actually
  slightly downwardly biased due to the fact that the training set under CV is
  slightly smaller than the training set to be used in actual deployment of
  the model.

- Leave-one-out cross validation (LOOCV) is cross validation with fold sizes
  of 1. In the case of linear models fit with ordinary least squares, the
  LOOCV prediction residuals can be obtained analytically as $r_i/(1-P_{ii})$,
  where $r_i$ is the residual and $P_{ii}$ are the diagonal elements of the
  projection matrix onto the column-space of the design matrix $X$.

- In most cases, LOOCV is the ideal form of cross validation, but is expensive
  if the model must be refit for each fold. In the case of many types of least
  squares fits, the prediction residuals can be calculated analytically, so
  the LOOCV result (often known as "PRESS", for prediction residual error sum
  of squares), can be calculated without ever refitting the model.

- The PRESS (or LOOCV) statistic for ordinary least squares is
  $\sum_i r_i^2/(1-P_{ii})$, where $r_i = \hat{Y}[i] - Y_i$. As an
  approximation, rather than dividing each residual by its corresponding
  $1 - P_{ii}$, we can replace each term of the form $1 - P_{ii}$ with their
  average, which is $1 - {\rm tr}(P)/n$. For models fit with OLS,
  ${\rm tr}(P) = p$, the number of covariates, so the LOOCV statistic becomes
  $(1-p/n)^{-1}\sum_i r_i^2$. This technique is known as _generalized cross
  validation_.

  - Since $(1-p/n)^{-1}$ increases with $p$ and decreases with $n$, we can see
    that the net effect of LOOCV is to inflate the plug in sum of squared
    residuals $\sum_i r_i^2$ by a factor that depends on the model complexity,
    and the amount of data available to accommodate that complexity.

- The _degrees of freedom_ aims to capture in one number the flexibility or
  complexity of a class of models. In least square regression, the degrees of
  freedom is simply the number of covariates.

- Many types of predictive models have _effective degrees of freedom_ that
  differ from (usually being lower than) the number of parameters. For
  example, if the model is fit with regularization, this will reduce the
  effective degrees of freedom. For example, in the case of ridge regression
  where we penalize the residual sum of squares with a quadratic form
  $\lambda\beta^\prime D \beta$, the effective degrees of freedom is
  ${\rm tr}[(X^\prime X + \lambda D)^{-1}X^\prime X]$.

- Cross-validation unbiasedly assesses the predictivity of a model on an
  independent sample from the same population as was used to fit the model. If
  the data-generating model is not-stationary, CV will not be able to account
  for the resulting change in predictive accuracy.

- An alternative to cross-validation is "true validation" in which a single
  independent training set is available. In this case, the model is fit once
  on the training set and then evaluated on the test set. In this case, the
  test set can, if appropriate, be sampled from a different population as the
  training set to assess generalization performance under distribution shift,
  or even due to a change in the population regression function $E[Y|X]$.

## Model selection

- Cross-validation can be used to estimate the predictivity of a model (or
  more specifically, of the algorithm used to fit the model). Sometimes this
  is the goal in itself. But in other settings the predictivity (e.g. $R^2$ or
  Brier score) is used to select from among a set of candidate models. The
  primary goal is not to estimate the predictivity, but to identify the
  best-performing model.

- Cross-validation is one of several ways to carry out model selection. When
  used for this purpose, rather than selecting the model with best
  predictivity, it is common to select the simplest model whose predictivity
  is within one standard-deviation of the best predictivity. This is known as
  the "one standard deviation rule". Utilizing this approach requires that we
  have a means to estimate the standard error of the predictivity estimates.

## Interpreting models

- Simple regression models were traditionally interpreted by direct inspection
  of the model parameters. For example, if the model is based on a linear
  predictor, each predictor variable has a "slope" that can be used to assess
  the contribution of the predictor variable to the prediction.

  - Suppose that we model the yield of a chemical reaction $Y$ in terms of
    pressure $P$ and temperature $T$, using a linear model
    $E[Y | P, T] = b_0 + b_1T + b_2P$. The coefficients are directly
    interpretable and even have interpretable units -- if yield is measured in
    milligrams (mg), pressure is measured in decibars (dbar),and temperature
    is measured in degrees centigrade, then $b_1$ is the change in average
    yield (in milligrams) associated with a one degree increase in
    temperature, while holding pressure fixed. The units of $b_1$ are
    milligrams per degree centigrade. Also, we can interpret $b_1$ as the
    derivative of $E[Y | P, T]$ with respect to $T$.

  - Now suppose that we consider a more general class of models of the form
    $E[Y | P, T] = bT + g(P)$. This is a _partially linear_ model (also an
    _additive model_). The function $g$ is unknown and need not be linear. In
    practice, we will need to put some constraints on $g$ such as requiring it
    to be smooth or monotone. Note that $b$ remains the derivative of expected
    yield ($E[Y | P, T]$) with respect to temperature ($T$), while holding
    pressure fixed, and continues to have units milligrams per degree
    centigrade. Since $g^\prime$ need not be constant, there is no fixed rate
    of change of expected yield with respect to pressure ($P$).

  - Now consider an even more general class of models of the form
    $E[Y | P, T] = b_1T + b_2TP + g(P)$. The rate of change of expected yield
    with respect to temperature is $b_1 + b_2P$. That is, the rate of change
    of expected yield with respect to temperature, holding pressure fixed, now
    depends on the specific value where we fix the pressure. This is a
    non-additive model, where the non-additivity results from the interaction
    between pressure and temperature.

- The examples above illustrate that modern models fit to high dimensional
  predictor data, or employing complex non-linear forms such as basis
  functions and interactions, are not amenable to direct interpretation. A
  number of techniques have been devised to make such models more
  interpretable.

- Suppose that we have an outcome $Y$, a "focus predictor" $X$, and other
  predictors $Z$. For simplicity, here we take the focus predictor to be a
  scalar, but the "other predictors" in $Z$ can be multidimensional. Our goal
  is to understand the role of $X$ in the regression function $E[Y | X, Z]$.

- One set of techniques focuses on the contribution of $X$ to the overall
  prediction of $Y$. If $R_{xz}^2$ is the $R^2$ (coefficient of determination)
  when regressing $Y$ on $X$ and $Z$, and $R_z^2$ is the $R^2$ when regressing
  $Y$ on $Z$ alone, then the partial $R^2$ is
  $(R_{xz}^2 - R_z^2) / (1 - R_z^2)$.

- When fitting models using ordinary least squares, the partial $R^2$ is equal
  to the usual $R^2$ for a regression involving residuals. Specifically, we
  regress both $Y$ and $X$ on $Z$, take residuals for each regression, then
  regress these residuals on each other. If we present the scatterplot of
  these residuals, rather than focusing on the correlation summary statistic,
  we have an _added variable plot_.

- The partial $R^2$ deals elegantly with the fact that the explanatory
  variables are likely to be related to each other. When we residualize the
  focus variable $X$ against $Z$, we are left with the unique information in
  $X$ that is not also present in $Z$.

- A very simple way to visually capture the contribution of $X$ to the
  variation in $Y$ is by plotting the graph of $f(X, EZ)$, viewed as a
  function of $X$, where $E[Z]$ is the average value of $Z$. This is an easy
  graph to construct but it is important to remember that in a non-additive
  model the relationship between $Y$ and $X$ generally depends strongly on
  $Z$, and fixing $Z$ at its expected value captures only part of the overall
  relationship of interest.

- A more systematic way to visualize the contribution of a variable to a
  fitted model is to consider the role of $X$ at each observed value of $Z$.
  This leads us to an _individual conditional expectations_ (ICE) plot.
  Suppose we fit a model $f(X, Z)$ to our data. For a given observation $i$,
  $f(X, Z_i)$, viewed as a function of $X$, captures the relationship between
  $Y$ and $X$ when holding $Z$ fixed at $Z_i$. A conventional ICE plot
  displays all $n$ of these curves on the same graph.

  - In an additive model, where we can write $f(X, Z) = g(X) + h(Z)$, the
    curves displayed in the ICE plot are parallel to each other, and it is
    easy to perceive the common features of these curves.

- A _partial dependence plot_ (PDP) is a simplified version of an ICE plot, in
  which the simplification results from a marginalization operation. It is
  simply the pointwise mean of the curves in the ICE plot. That is, the PDP,
  plotted at $X$, is the average value of the $f(X, Z_i)$, over all $Z_i$ in
  the dataset.

- Mediation analysis is a tool for understanding regression models in which
  there are plausible causal relationships among the predictors. Suppose we
  have an _exposure_ $X$, a _mediator_ $M$, and an outcome $Y$. The goal is to
  understand how changes in $X$ associate with changes in $Y$, but it is
  considered likely that changes in $X$ also associate with changes in $M$.
  Since changes in $M$ also associate with changes in $Y$, this opens up to
  pathways by which $X$ can impact $Y$ -- a _direct effect_ that bypasses $M$,
  and an _indirect effect_ that flows through $M$.

- As an example, suppose we have a drug that is intended to lower serum
  cholesterol, and thereby produce an outcome such as vascular calcification.
  Cholesterol can be measured, and plays the role of the mediator ($M$) here.
  Whether a person takes the drug is the exposure ($X$), and vascular
  calcification is the outcome ($Y$). We do not know for certain whether there
  are other mechanisms by which the drug can impact calcification (other than
  by lowering cholesterol). Mediation analysis can be used to partition the
  effect of the drug ($X$) into the part that is due to lowering cholesterol
  (the mediated effect) and the part that is unrelated to lowering cholesterol
  (the direct effect).

- Formally, mediation analysis proceeds by considering _counterfactual_
  outcomes -- let $M(1)$ denote the mediator when $X=1$ and $M(0)$ denote the
  mediator when $X=0$. These are also known as _potential outcomes_, and for
  any individual, only one of $(M(0), M(1))$ is observed. Note also that
  $(M(0), M(1))$ is random and varies across the cases. Let $Y(X, M(x))$
  denote a (random) observation made at exposure level $X$ and mediator level
  $M(x)$. This observation is counterfactual since $X$ need not equal $x$.

- The direct effect can be defined as:
  $E[Y | X=1, M=M(0)] - E[Y | X=0, M=M(0)]$

- The indirect effect can be defined as:
  $E[Y | X=0, M=M(1)] - E[Y | X=0, M=M(0)]$

- The total effect can be defined as:
  $E[Y | X=1, M=M(1)] - E[Y | X=0, M=M(0)]$

- There are various ways to estimate mediation effects, but we will not
  attempt to cover these methods in detail here. The main idea is to fit
  regression models for $Y|M, X$ and for $M|X$, then simulate counterfactual
  values from $M|X$ using the model, and use the model for $Y|M, X$ to
  estimate all the needed expectations.
