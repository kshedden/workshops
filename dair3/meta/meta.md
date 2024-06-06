# Meta-analysis

## Overview of research synthesis

- [Research synthesis](https://en.wikipedia.org/wiki/Research_synthesis) is a
  form of research that combines results from multiple previously completed
  research studies.

- [Meta analysis](https://en.wikipedia.org/wiki/Meta-analysis) is a specific
  type of research synthesis in which the aim is to quantify a specific
  parameter of interest, using results from multiple sources reflecting
  well-conducted
  [quantitative](https://en.wikipedia.org/wiki/Quantitative_research) research
  studies.

- Pooling research results almost always leads to more
  [precise](<https://en.wikipedia.org/wiki/Precision_(statistics)>) estimates.
  In some cases, [bias](https://en.wikipedia.org/wiki/Bias_of_an_estimator) is
  reduced by meta analysis but this is not always the case.

- A [systematic review](https://en.wikipedia.org/wiki/Systematic_review) is
  another type of research synthesis that we will not discuss further here. It
  integrates [qualitative](https://en.wikipedia.org/wiki/Qualitative_research)
  research studies while our focus here is on integration of quantitative
  research findings. Further, meta-analyses are usually focused on a specific
  quantitative aim (estimation or hypothesis testing), whereas systematic
  reviews are often more open-ended.

- Research synthesis can be used to consolidate and unify findings that are
  mostly consistent with each other, but can also reveal novel insights and
  suggest alternative theories.

## Pooling estimates asnd standard errors

- Suppose we have estimates $\hat{\theta}_1$ and $\hat{\theta}_2$, say of the
  association between an exposure and an outcome, obtained from two independent
  research studies. The corresponding
  [standard errors](https://en.wikipedia.org/wiki/Standard_error) are $s_1$ and
  $s_2$.

- We will _pool_ the estimates, first by simple averaging to obtain
  $\hat{\theta}_p \equiv (\hat{\theta}_1 + \hat{\theta}_2)/2$.

- A fundamental fact that underlies a lot of meta analysis is that the standard
  error of $\hat{\theta}_p$ is $\sqrt{(s_1^2 + s_2^2)/2}/\sqrt{2}$.

  - The logic behind this fact is that variances add, so
    ${\rm Var}(\hat{\theta}_1 + \hat{\theta}_2) = s_1^2 + s_2^2$. Further, the
    variance of the mean is half of the average variance, and the standard
    deviation of the mean is $1/\sqrt{2}$ times the average variance.

  - We can generalize this result to allow weights. Let $w_1$ and $w_2$ denote
    weights and define
    $\hat{theta}_p = (w_1\hat{\theta}_1 + w_2\hat{\theta}_2) / (w_1 + w_2)$.
    Then the standard error of $\hat{\theta}_p$ is

$$\sqrt{w_1^2s_1^2 + w_2^2s_2^2}/(w_1 + w_2)$ = \sqrt{(w_1^2s_1^2 + w_2^2s_2^2) /  (w_1^2 + w_2^2)} \cdot \sqrt{(w_1^2 + w_2^2)/(w_1 + w_2)^2}.$$

- The factor $\sqrt{(w_1^2 + w_2^2)/(w_1 + w_2)^2}$ is a measure of the gain in
  precision from pooling. Note that if the weights are equal, $w_1 = w_2 = 1$,
  then this factor is equal to $1/\sqrt{2}$ as in the special case of simple
  averaging.

## Data integration

- In a conventional meta-analysis, we only have access to summary data from the
  individual studies (usually extracted from publications). What if we have
  access to some or all of the raw data from the individual studies?

-

## Study selection

- "Garbage in garbage out"

- No universal rules, but consider the following

  - Do the outcome and exposures match your research aims?

  - Year of publication

  - Handling of confounding factors

  - Adequate information provided in the publication (or made available by
    authors)

  - Generally exclude studies that are themselves research syntheses

  - Multiple studies of the same subject pool

## Integration of p-values

- The most conventional way to conduct statistical inference in a formal
  research study is a
  [statistical hypothesis test](https://en.wikipedia.org/wiki/Statistical_hyopothesis_test).
  This is a formalized process in which a
  [null hypothesis](https://en.wikipedia.org/wiki/Null_Hypothesis) is specified
  and we quantify evidence against the null hypothesis using a
  [test statistic](https://en.wikipedia.org/wiki/Test_statistic).

- The [p-value](https://en.wikipedia.org/wiki/P-value) is the most familiar
  method for summarizing evidence against a null hypothesis. Statistical
  inference that focuses especially on the p-value is known as
  [significance testing](https://en.wikipedia.org/wiki/Statistical_significance).

- A smaller p-value corresponds to stronger evidence against the null
  hypothesis. Conventionally a p-value threshold is imposed such as deeming
  $p\le 0.05$ to be significant. However this _dichotomization of evidence_ is
  controversial and any particular threshold is usually arbitrary.

- In the context of

## Heterogeneous effects

- Suppose we have $m$ studies, all aiming to estimate a treatment effect. Let
  $\theta_i$ denote the true treatment effect for the $i^{\rm th}$ study.

- In some cases, it may be reasonable to presume that there is a single common
  treatment effect, that is, all of the $\theta_i$ are equal to a common value
  $\theta$.

- Let $\hat{\theta}_i$ be the estimated treatment effect for the $i^{\rm th}$
  study. Even if there is only a single common treatment effect $\theta$, the
  $\hat{\theta}_i$ will nevertheless vary due to
  [sampling variation](https://en.wikipedia.org/wiki/Sampling_error).

- It is almost always impossible to exactly replicate a research study. Even
  when using a common study protocol, two studies will inevitably differ in
  terms of how the design is implemented. In research with human subjects, study
  populations almost inevitably change based on location or over time,
  ascertainment criteria can be interpreted in slightly different ways by
  different implementation teamns, and social and environmental confounders will
  generally vary from one study to another, even if identical protocols are
  followed.

## Heterogeneous designs

## Reporting biases
