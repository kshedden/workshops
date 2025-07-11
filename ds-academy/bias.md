# Identifying and compensating for sources of bias

## Overview

- There are a number of reasons that a data analysis may be subject to bias.
  We discuss a few reasons for bias and discuss possible remedies below.

## Biased samples

- Suppose we sample data from a distribution $Q$, but our goal is to make
  inferences with respect to a distribution $P$. Specifically, suppose we are
  interested in $\theta = E_P h(X)$ for a given function $h$.

- If we sample $x_1, \ldots, x_n$ from $Q$ and average them, the result will
  be biased.

- If $P$ and $Q$ are known, we can weight an observed value $x$ with weight
  $w(x) = P(x)/(n\cdot Q(x))$ to compensate for the bias. The estimate
  $\sum_i w(x_i)x_i$ is an unbiased estimate of $\theta$.

- The variance of the weighted estimator is $E[h(X)^2(dP(x)/dQ(x) - 1)]$

- Sometimes biased sampling is deliberate -- we define the distribution $Q$ to
  make the sampling more convenient, or to improve inference precision.

- Sometimes biased sampling is imposed on us -- e.g. due to non-response or
  being forced to work with a convenience sample.

## Missing data in regression

- Suppose we are analyzing data of the form $Z = (Y, X)$, where $Y$ and $X$
  are vectors of characteristics. The variables in $Y$ may be missing, but the
  variables in $X$ are always observed.

- Let $R$ denote the indicator vector showing which elements of $Y$ are
  observed, i.e. $R[j] = 1$ if $Y[j]$ is observed and otherwise $R[j] = 0$.
  Note that $R$ is data, and is always observed.

- Let $Y^c$ denote the complete value of $Y$, not subject to missingness.

- We can characterize the _missingness mechanism_ based on the statistical
  dependence of $R$, $X$, and $Y^c$.

  - If $R$ is independent of $(X, Y^c)$ the data are _missing completely at
    random_ (MCAR).

  - If $R$ is independent of $Y^c$ given $X$ the data are _missing at random_
    (MAR).

  - Otherwise the data are _missing not at random_ (MNAR).

- Since the definitions of missingness mechanisms depend on $Y^c$, the
  observed data alone cannot determine the missingness mechanism.

## Methods to analyze missing data

- A simple approach is _complete case analysis_, or _listwise deletion_. This
  means that we drop all $Z$ for which any component is missing.

  - If the data are missing completely at random, complete case analysis
    yields unbiased results.

  - If the data are missing at random, and our analysis involves the
    regression function $E[Y|X]$, complete case analysis yields unbiased
    results.

  - Even when there is no bias, complete case analysis can be suboptimal in
    terms of efficiency, since we do not utilize information in $Z$ for any
    case with even a single missing component.

- Full Information Maximum Likelihood (FIML) is an approach that works for
  model-based analyses. If $I_o$ indexes the positions in $Y$ that are
  observed and $I_m$ indexes the positions in $Y$ that are missed, we can use
  $\int p(Y[I_o], Y[I_m] | Z)dY[I_m]$ as the likelihood for an observed case,
  and proceed, e.g. using maximum likelihood analysis.

- FIML is unbiased as long as the data are missing at random (MAR).

- The Expectation-Maximization (EM) algorithm is an approach for calculating
  the maximum likelihood estimate when direct marginaliation is difficult.

  - Start with a parameter value $\eta$

  - In the "E step" we calculate
    $Q(\theta) = E_\eta[\log p_\theta(Y | Z) | Y[I_o]]$

  - In the "M step" we maximize $Q(\theta)$ to obtain a new value of $\eta$

  - The EM algorithm is a special case of a more general class of
    majorization/maximization (MM) algorithms. The main practical advantage of
    the MM algorithm over direct maximization is that it allows us to work
    with the log of the data generating density.

- Multiple imputation (MI)

  - Suppose we have a model $Q_\theta$ for our data. We can fit the model to
    the observed data yielding a parameter estimate $\hat{\theta}$.

  - Next, impute $m$ complete datasets, by taking each $X$ and imputing
    $X[I_m]$ with a sample from $X[I_m] | X[I_o]$, with the sample drawn from
    $Q_\hat{\theta}$.

  - For the $j^{\rm th}$ completed dataset, estimate the parameter of interest
    $\hat{\eta}[j]$ and obtain its sampling variance $V[j]$.

  - Average the $\eta[j]$ to obtain a point estimate for $\eta$.

  - Apply the _combining rule_
    ${\rm var}(\eta[1], \ldots, \eta[m]) + {\rm avg}(V_1, \ldots, V_m)$ to
    obtain the sampling variance of the point estimate.

  - Ideally, each imputed dataset is sampled from a $Q_{\tilde{\theta}}$,
    where $\tilde{\theta}$ is a draw from either the parameteric or
    nonparametric bootstrap sampling distribution of $\theta$.

  - MI is unbiased as long as the data are missing at random (MAR).

- Multiple Imputation via Chained Equations (MICE)

  - Given a data vector $X$, propose working models for the conditional
    distribution of each component of $X$ given the other components; i.e.
    propose models for $X[j] | X[-j]$

  - Cycle over the variables $j$, and update the working model using only the
    cases for which $X[j]$ is observed. Then sample the $X[j]$ for the
    non-observed cases from the working model.

  - Retain a collection of completed datasets.

  - The samples are analyzed using the same combining rule as MI.

  - MICE is a practical alternative to Gibbs sampling, in which the
    conditional distributions used for sampling are derived from a single
    parent distribution $P$. In the case of MICE, the distributions used for
    sampling may not be the conditional distributions for any single
    distribution.

## Covariate measurement error

- In a regression analysis, we are interested in $E[Y|X]$, and it is nearly
  always the case that $Y$ is subject to random variation around its mean,
  i.e. ${\rm var}[Y|X] > 0$. This random variation could be measurement error
  as well as intrinsic stochastic variation in the value of interest.

- In most regression analyses, it is assumed that the covariates $X$ are
  observed without error.

- If $\psi$ is the variance/covariance matrix of the covariate measurement
  errors, and $\Sigma_x$ is the variance/covariance matrix of the covariates,
  the bias in a least squares regression due to covariate measurement error is
  $((I + \Sigma_x^{-1}\Psi)^{-1} - I)\beta$.

- $(I + \Sigma_x^{-1}\Psi)^{-1}$ is a contraction, so the bias has the effect
  of shrinking the norm of $\beta$. Thus, covariate measurement error leads to
  attenuation on average, but not necessarily in every coefficient.

## Confounding

- Suppose we are interested in the causal effect of a treatment $T$, which we
  take here for simplicity to be binary ($T=0, 1$).

- Let $Y$ denote the treatment response.

- If assignment to treatment is random, we can estimate the treatment effect
  as ${\rm Avg}(Y|T=1) - {\rm Avg}(Y|T=0)$.

- If assignment to treatment is not random, it may be associated with a
  covariate $X$ that also affects $Y$. In this case, the naive estimate of the
  treatment effect will be biased.

- If $X$ is observed, we can model $P(Y|T, X)$, and if the model is correctly
  specified, the model-based treatment effect estimates will be free of
  confounding from $X$.

- The _propensity score_ is the probability $P(T|X)$. If we can estimate the
  propensity score, then its reciprocal can be used as a weight to eliminate
  confounding by $X$.

- Alternatives to using the propensity score as a weight are to match or
  stratify on the propensity score.

  - Some people prefer to match on the covariate $X$ directly rather than on
    the propensity score.
