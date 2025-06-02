# Model interpretation and causal inference

## Predictivity

- We will say that a model for a quantitative outcome $Y$ is predictive if it
  explains a portion of the variation of $Y$. If we consider the
  [law of total variation](https://en.wikipedia.org/wiki/Law_of_total_variance),
  the variance of $Y$ can be decomposed as
  ${\rm Var}[Y] = {\rm Var} E[Y|X] + E {\rm Var}[Y|X]$, where $X$ is a vector
  of predictor variables. Since both ${\rm Var} E[Y|X]$ and $E {\rm Var}[Y|X]$
  must be non-negative, we can interpret ${\rm Var} E[Y|X]$ as the variation
  of $Y$ that is _explained_ by the predictors $X$.

- Most useful models need to be predictive to some extent. That is, in almost
  all circumstances, a useful model should satisfy ${\rm Var} E[Y|X] > 0$. For
  the sake of completeness, in some cases it may be sufficient for a model to
  explain some of the
  [heteroscedasticity](https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity)
  in $Y$, even while explaining none of the variation in $Y$ (i.e.
  ${\rm Var} E[Y|X] = 0$) but we will not consider this setting further here.

### Measures of predictivity

- The amount of variation of $Y$ that is explained by $X$ can be quantitified
  using the
  [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination),
  also known as the $R^2$, which can be defined as
  ${\rm Var}E[Y|X]/{\rm Var}[Y]$. The $R^2$ ranges from $0$ to $1$. This is
  the _population_ $R^2$, which captures the total information in $X$ about
  $Y$.

- If we have a prediction $\hat{Y}$ of $Y$, we can consider
  ${\rm Var}[\hat{Y}]/{\rm Var}[Y]$ or ${\rm corr}(\hat{Y}, Y)^2$ (when
  $\hat{Y}$ is obtained using
  [ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares)
  the two are identical). This is a measure of how well the model used to
  produce $\hat{Y}$ captures the true value $Y$. Below we will consider some
  important practical issues associated with estimating this quantity
  [unbiasedly](<https://en.wikipedia.org/wiki/Bias_(statistics)>).

- The value of $R^2$ that can be considered "useful" is highly
  context-specific. In many applications involving human behavior, or
  considering individual factors within large systems (e.g. contributions of
  genes to phenotypes), $R^2$ values as small as 1%% or less can still be
  useful and meaningful.

- Besides the $R^2$, other measures of predictivity are useful in specific
  settings, especially when the variable $Y$ being modeled is not
  quantitative.

- When working with a binary response $Y$, and its predicted value $\hat{Y}$,
  it is useful to consider the following four conditional probabilities:

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
  predictive errors. A _false positive_ occurs when $\hat{Y}=1$ but $Y=0$, and
  a _false negative_ occurs when $\hat{Y} = 0$ but $Y=1$.

- Many models for a binary outcome $Y$ produce a quantitative score $Z$ rather
  than a binary decision. In this setting, a decision can be made by setting a
  threshold $t$, and predicting $Y=1$ when $Z>t$ and $Y=0$ when $Z\le t$. In
  this setting, higher values of $t$ result in increased specificity and
  decreased sensitivity. The
  [receiver operating characteristics curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic),
  or _ROC curve_ is a plot of sensitivity (on the vertical axis) against the
  complement (1 minus) the specificity (on the horizontal axis).

- The ROC curve is non-decreasing, and in general should lie above the
  diagonal $y=x$. If the ROC curve falls mostly or entirely below the line
  $y=x$, the model is predicting worse than guessing. The greater the value of
  the ROC curve, i.e. the higher the sensitivity at a given value of 1 minus
  specificity, the better the performance of the model. High values of the ROC
  curve toward the left side of the plot reflect better performance in very
  conservative (high specificity) regimes, and high values toward the right
  side of the plot reflect better performance in very permissive (low
  specificity) regimes.

- The area under the ROC curve is known as the AUC or the AUROC and is an
  important metric for the predictive performance of models for binary
  outcomes. The AUC ranges from 0 to 1, with higher values corresponding to
  better predictive performance.

- Insight into the AUC can be gained through its relationship to the
  (Mann-Whitney U
  statistic\](https://en.wikipedia.org/wiki/Mann-Whitney_U_test). In this way,
  we can see that the AUC is equal to $P(Z_1 > Z_2 | Y_1=1, Y_2=0)$, where the
  probability is computed over all independent pairs such that $Y_1=1$ and
  $Y_2=0$. As above, the $Z_j$ are quantitative scores reflecting the evidence
  that $Y=1$. For simplicity, here we consider that the $Z$'s are never tied.

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
  which refers to any situation where the population we are observing drifts
  over time.

- The presence of non-stationarity usually degrades the _generalization_
  performance of models, and may also impact their basis for making causal
  inferences.

- An extreme form of non-stationarity forces us to extrapolate, which occurs
  when the support of $P^{\rm test}_X$ is different from the support of
  $P^{\rm train}_X$. This means that there are regions with positive
  probability under $P^{\rm test}_X$ that have zero probability of occuring
  under $P^{\rm train}_X$.

- Distribution shift refers to the situation where the marginal distribution
  of predictors $P_X$ changes over time. Another form of non-stationarity
  occurs when the probability distribution $P(Y|X)$ or more narrowly the
  conditional mean $E[Y|X]$ changes over time. This form of non-stationarity
  is even more challenging that what is encountered under distribution shift,
  since we now cannot know whether relationships among variables of interest
  are stable over time.

## Estimating predictivity

-
