# Meta-analysis

## Overview of research synthesis

- [Research synthesis](https://en.wikipedia.org/wiki/Research_synthesis) is a
  form of [research](https://en.wikipedia.org/wiki/Research) that combines
  results from multiple previously completed research studies.

- [Meta analysis](https://en.wikipedia.org/wiki/Meta-analysis) is a specific
  type of research synthesis in which the aim is to quantify a specific
  parameter or test a specific hypothesis of interest, using results from
  multiple [quantitative](https://en.wikipedia.org/wiki/Quantitative_research)
  sources, and objectively assessing uncertainty for all claims.

- Pooling research results almost always leads to more
  [precise](<https://en.wikipedia.org/wiki/Precision_(statistics)>)
  estimation, and to a better understanding of effect
  [moderators](<https://en.wikipedia.org/wiki/Moderation_(statistics)>). In
  some cases, [bias](https://en.wikipedia.org/wiki/Bias_of_an_estimator) is
  reduced by meta analysis but this is not always the case.

- There are both practical and deeply conceptual aspects of meta analysis.

  - At a practical level, we are concerned with understanding the real-world
    context of research, in which the quality of individual sources of
    evidence can be affected by researcher behavior, the culture of rigor in
    different fields, blindspots and sensitivities about threats to rigor in
    different fields, resource constraints, and technical capabilities and
    limitations for data collection and measurement.

  - Assuming that we can quantify the quality (normally operationalized
    through _bias_ and _precision_ of individual sources of evidence), we then
    need to consider how to formally and quantitatively integrate this
    information into a holistic finding that reflects the overall state of
    knowledge. This turns out to be extremely challenging, especially since in
    practice there is not only uncertainty about the results of any given
    evidence source, but there is also uncertainty about the level of
    uncertainty.

- A [systematic review](https://en.wikipedia.org/wiki/Systematic_review) is
  another type of research synthesis that is not our focus here. A systematic
  review summarizes a set of related findings, but does not aim to estimate a
  specific quantitative target by pooling evidence. Also, a systematic review
  may integrate both quantitative and
  [qualitative](https://en.wikipedia.org/wiki/Qualitative_research) research
  studies, while meta-analyses usually consider only quantitative studies.

- Research synthesis can be used to consolidate and unify findings that are
  mostly consistent with each other, but can also reveal novel insights and
  suggest alternative theories.

- The
  [Cochrane organization](<https://en.wikipedia.org/wiki/Cochrane_(organisation)>)
  is highly influential in advocating for and serving as a repository for high
  quality research synthesis in clinical care including meta-analyses.

- The [PRISMA statement](https://prisma-statement.org) provides guidance for
  meta analyses that many investigators aim to follow.

## Pooling estimates and standard errors

- In many settings where meta-analysis is applied, the focus is on a single
  quantitative parameter $\theta$. We have multiple estimates of $\theta$ from
  different studies, which we write $\hat{\theta}_1$, $\hat{\theta}_2$,
  $\ldots$. In addition we have estimates of uncertainty about each of these
  estimates which usually take the form of
  [standard errors](https://en.wikipedia.org/wiki/Standard_error), and we will
  denote them $\hat{s}_1$, $\hat{s}_2$, $\ldots$.

- It is important to note that the standard errors are almost always
  themselves estimates, which is why we put a "hat" over them. In some cases,
  we have one more layer of evidence which takes the form of
  [degrees of freedom](<https://en.wikipedia.org/wiki/Degrees_of_freedom_(statistics)>)
  $k_1$, $k_2$, $\ldots$. The degrees of freedom quantifies the confidence
  that we have in the standard errors, with greater degrees of freedom
  corresponding to greater confidence. If the degrees of freedom $k_i$ for the
  estimator $\hat{\theta}_i$ is infinite, then the standard error $\hat{s}_i$
  can be considered to be known exactly.

- Let's consider the special cas of a meta-analysis of two studies, which
  produed estimates $\hat{\theta}_1$ and $\hat{\theta}_2$, with standard
  errors $s_1$ and $s_2$. For now, consider the degrees of freedom to be
  infinite so we omit the hats over $s_1$ and $s_2$ to convey that they are
  exact.

- We can _pool_ the estimates using unweighted or weighted averaging.

- One version of a pooled estimate is the simple average of the estimates,
  $\hat{\theta}_a = (\hat{\theta}_1 + \hat{\theta}_2)/2$.

- The standard error of this simple pooled estimate is
  $\sqrt{(s_1^2 + s_2^2)/2}/\sqrt{2}$. The factor $\sqrt{(s_1^2 + s_2^2)/2}$
  is the standard deviation derived from the average variance. The factor of
  $1/\sqrt{2}$ is the improvement in precision that results from pooling
  evidence from two independent studies.

- The optimal (most efficient) pooled estimator is
  [weighted](https://en.wikipedia.org/wiki/Inverse-variance_weighting) by the
  inverse variances. That is, we weight the first study by $1/s_1^2$ and the
  second study by $1/s_2^2$. The resulting pooled effect estimate is
  $\hat{\theta}_w = (\hat{\theta}_1/s_1^2 + \hat{\theta}_2/s_2^2) / (1/s_1^2 + 1/s_2^2)$.

- If the standard errors are equal, i.e. $s_1 = s_2$, then the unweighted and
  weighted pooled estimators are the same.

- The standard error of the inverse variance weighted average is
  $1 / (1/s_1^2 + 1/s_2^2)^{1/2}$. Letting $H$ denote the
  [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean) of the
  study-level variances, then the standard deviation of the inverse variance
  weighted average is $H^{1/2}/\sqrt{2}$.

- If we are pooling $m$ independent estimates, the standard error of the
  inverse variance weighted average is $H^{1/2}/\sqrt{m}$, where $H$ is the
  harmonic mean of the $k$ study-level variances.

- Things become much more complicated when the $s_i$ are not known exactly
  (i.e. when the degrees of freedom are finite). In this case, it is still
  common to use the inverse variance weighted mean as a consensus estimate,
  using plug-in weights $\hat{s}_i^{-2}/\sum_j \hat{s}_j^{-2}$. But in this
  case it becomes much more difficult to quantify the uncertainty of the
  pooled estimate.

  - The standard error below approximately captures the inflation of
    uncertainty due to estimation of the weights:

  $\sqrt{H(1 + 4n^2H^{-1}\sum_j n_j^\prime w_j(H/n - w_j))} / \sqrt{n}$

  This estimate can be taken to have degrees of freedom

  $n^{-2}H^2 / \sum_j w_i^2/k_j$.

  - It is difficult to account for possible statistical dependence between
    $\hat{\theta}_i$ and $\hat{s}_i$.

  - Since many statistics are asymptotically normal, it is common to treat the
    $\hat{\theta}_i$ as following normal distributions. However some
    statistics converge very slowly to their limiting distribution. An example
    of this would be the rate of occurrence of a rare event.

  - If a
    [variance-stabilzing transformation](https://en.wikipedia.org/wiki/Variance-stabilizing_transformation)
    is available, then it may be advisable to pool evidence using the
    transformed estimates.

  - An emerging approach that can work well with smaller meta-analyses is
    _jackknife empirical likelihood_ analysis.

## Data integration

- In a conventional meta-analysis, we only have access to summary data from
  the individual studies (usually extracted from publications). What if we
  have access to some or all of the raw data from the individual studies? This
  can be referred to as an _individual patient data meta-analysis_.

- We can pool the data and re-analyze it as a single data set, but it is
  important to be aware of heterogeneity (systematic differences) among the
  studies.

- Pooling data from multiple studies raises similar issues as arise in a
  single study in which the data are drawn from heterogeneous clusters (e.g. a
  multi-center research trial).

- There are many ways to account for study heterogeneity in an integrated
  analysis. Here are a few popular approaches:

  - A regression analysis can employ
    [fixed effects](https://en.wikipedia.org/wiki/Fixed_effects_model) for the
    different studies being pooled.

  - A multilevel regression analysis can employ random effects for the
    different studies being pooled. Random effects require somewhat stronger
    assumptions to be effective (compared to fixed effects), but typically
    provide better statistical power.

  - There are many forms of _stratified analysis_. These approaches
    essentially conduct independent analyses in each study and pool the
    results. The generic stratified analysis pools parameter estimates and
    standard errors as discussed above.

  - In some settings, special-purpose stratified analysis methods have been
    devised, one of these is the
    [Mantel-Haenszel test](https://en.wikipedia.org/wiki/Cochran-Mantel-Haenszel_statistics)
    for integrating contingency tables. This approach essentially pools the
    [odds ratios](https://en.wikipedia.org/wiki/Odds_ratio) from multiple
    studies.

  - Random effects approaches conduct a form of _partial pooling_, which is a
    compromise between random effects, fixed effects, and naive analysis (no
    pooling).

## Study selection

- [Garbage in garbage out](https://en.wikipedia.org/wiki/Garbage_in_garbage_out)

- There are no universal rules, but consider the following

  - Do the outcome and exposures match your research aims?

  - Include both negative and positive findings

  - Reporting of quantitative findings (point estimates, standard errors)

  - Reporting of methods

  - Definition of target study population, and inclusion/exclusion criteria

  - Handling of confounding factors

  - Generally exclude studies that are themselves research syntheses

  - Multiple studies of the same subject pool are not independent

  - Multiple studies by the same research team may not be independent

  - Is it possible to include unpublished studies (pre-registration may make
    this possible)

  - Publication language

  - Select by date? Include all studies published after a specific date

## Integration of p-values

- The most conventional way to conduct statistical inference in a formal
  research study is a
  [statistical hypothesis test](https://en.wikipedia.org/wiki/Statistical_hyopothesis_test).
  This is a formalized process in which a
  [null hypothesis](https://en.wikipedia.org/wiki/Null_Hypothesis) is
  specified and we quantify evidence against the null hypothesis using a
  [test statistic](https://en.wikipedia.org/wiki/Test_statistic).

- The [p-value](https://en.wikipedia.org/wiki/P-value) is the most familiar
  method for summarizing evidence against a null hypothesis. Statistical
  inference that focuses primarily on the p-value is known as
  [significance testing](https://en.wikipedia.org/wiki/Statistical_significance).

- A smaller p-value corresponds to stronger evidence against the null
  hypothesis. Conventionally a p-value threshold is imposed such as deeming
  $p\le 0.05$ to be significant. However this
  [dichotomization of evidence](https://statmodeling.stat.columbia.edu/wp-content/uploads/2017/11/jasa_combined.pdf)
  is controversial and any particular threshold is usually arbitrary.

- There are several ways to integrate independent p-values, that is, to
  produce an overall p-value that combines the evidence in the individual
  p-values. This overall p-value reflects the _global_ or _omnibus_ null
  hypothesis that the null hypothesis is true in all studies being combined.

  - The fact that underpins most p-value combining procedures is that if the
    null hypothesis is true, the p-value follows a uniform distribution on the
    interval $(0, 1)$.

  - [Fisher's method](https://en.wikipedia.org/wiki/Fisher%27s_method)
    constructs a statistic $T = -2\sum_j \log(p_j)$, where $p_1, \ldots, p_m$
    are the independent p-values from the studies being analyzed via
    meta-analysis. Under the global null hypothesis, $T$ follows a
    [chi-squared](https://en.wikipedia.org/wiki/Chi-squared_distribution)
    distribution with $2m$ degrees of freedom. A p-value can then be
    constructed from $T$ based on this reference distribution.

  - Most p-values are derived from Z-scores, particularly when the p-value
    derives from a [Wald test](https://en.wikipedia.org/wiki/Wald_test) of a
    single parameter. If we have independent Z-scores $Z_1, \ldots, Z_m$, we
    can obtain a global p-value by first pooling the Z-scores to obtain
    _Stouffer's Z_, which is $Z_p = m^{-1/2}\sum_j Z_j$. Under the global null
    hypothesis, $Z_p$ has a standard normal distribution, and it is
    straightforward to obtain a p-value from $Z_p$ if desired.

- More recently, some progress has been made on the much harder problem of
  combining p-values that are not independent (e.g. from tests that may be
  correlated).

  - The classical method for non-independent p-values is the
    [Bonferroni method](https://en.wikipedia.org/wiki/Bonferroni_correction),
    which uses the value of $m\times {\rm min}(p_1, \ldots, p_m)$ as a meta
    p-value.

  - Work by [Vovk](https://arxiv.org/pdf/1212.4966) shows that the arithmetic,
    geometric, and harmonic mean p-values can be used as "meta p-values" if
    they are rescaled appropriately. Specifically,
    $2{\rm Avg}(p_1, \ldots, p_m)$, $e{\rm GM}(p_1, \ldots, p_m)$, and
    ${\rm log(m)}{\rm HM}(p_1, \ldots, p_m)$ can be used to assess the global
    null hypothesis, for the arithmetic, geometric, and harmonic means,
    respectively.

  - Under the
    [Cauchy combining rule](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7531765),
    the p-values are combined using $T = \sum_j \tan(\pi(1/2 - p_j))$. Under
    the global null hypothesis, $T$ follows a standard Cauchy distribution,
    and therefore can be transformed to a p-value using the CDF of the
    reference distribution.

- Sometimes it is desirable to to produce a "reject/accept" decision regarding
  a null hypothesis, rather than producing a meta p-value. One way to do this
  is using the
  [Holm Bonferroni stepdown method](https://en.wikipedia.org/wiki/Holm-Bonferroni_method).
  First, suppose that the p-values have been sorted in ascending order. If for
  any $j$, $p_j < \alpha / (m + 1 - j)$, the null hypothesis is rejected. This
  procedure controls the false positive rate for the global test at $\alpha$.

## Heterogeneous effects

- Suppose we have $m$ studies, all aiming to estimate a treatment effect. Let
  $\theta_i$ denote the true treatment effect for the $i^{\rm th}$ study.

- Let $\hat{\theta}_i$ be the estimated treatment effect for the $i^{\rm th}$
  study. Each $\hat{\theta}_i$ deviates from the corresponding true $\theta_i$
  due to [sampling variation](https://en.wikipedia.org/wiki/Sampling_error).

- In some cases, it may be reasonable to presume that there is a single common
  treatment effect, that is, all of the $\theta_i$ are equal to a common value
  $\theta$. This leads to the meta-analysis model
  $\hat{\theta}_i = \theta + s_i\epsilon_i$, where $\epsilon_i$ are
  unit-variance random deviations for each study. As discussed above, the
  value of $\theta$ is efficiently estimated using the inverse variance
  weighted mean of the individual study estimates, which we denote
  $\hat{\theta}$.

- It is rarely possible to perfectly replicate a research study. Even when
  using a common study protocol, two studies will generally differ in terms of
  how the design is implemented. Here are some reasons why heterogeneity may
  be present:

  - Study populations vary based on geographic location or over time in terms
    of comorbidities, genetic background, environmental risks, demographics,
    socioeconomic status, medication histories, and the severity and type of
    disease.

  - The treatment can be implemented differently in different locations or
    different historical periods.

  - Ascertainment (inclusion/exclusion) criteria can be interpreted in
    different ways by different implementation teams.

- To account for study heterogeneity, a common approach to meta-analysis is
  the _random effects_ or _hierarchical_ approach, in which we posit the model
  $\hat{\theta}_i = \theta_i + s_i\epsilon_i$, where the $\theta_i$ are
  treated as random variables from a distribution with mean $\theta$ and
  variance $\tau^2$, and the $\epsilon_i$ are also random variables that are
  independent of the $\theta_i$ and that follow a distribution with mean $0$
  and unit variance.

  - Sometimes we allow the $\epsilon_i$ to have variance $\sigma^2$, to
    account for systematic biases (e.g. inflation) in the reported standard
    errors.

- The parameters of the random effects model can be estimated using methods
  for fitting mixed models (maximum likelihood and restricted maximum
  likelilhood), but it is also possible to perform the estimation using
  Cochrane's Q-statistic, which was an early measure of study heterogeneity.

- The Q-statistic is defined as
  $Q \equiv \sum (\hat{\theta}_j - \hat{\theta}_p)^2 / \hat{s}_j^2$. If there
  is no heterogeneity (all $\theta_i$ are equal), $Q$ asymptotically follows a
  $\chi^2$ distribution with $m-1$ degrees of freedom under the null
  hypothesis of no heterogeneity (i.e. when $\tau^2=0$).

  - In principle, this allows $Q$ to be used to formally test a null
    hypothesis of no heterogeneity. However $Q$ is only asymptotically
    $\chi^2$ and it has long been known that the $\chi^2$ approximation is
    poor unless you have a large number of studies, and the studies themselves
    are large.

  - Some empirical work suggests that the null distribution of $Q$ might
    approximately follow a
    [gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution), of
    which the $\chi^2$ distribution is a special case. One approach is to use
    the [method of moments](https://en.wikipedia.org/wiki/Method_of_moments)
    to estimate the particular gamma distribution that best matches the null
    distribution, but this requires estimation of moments of $Q$ which is
    quite complicated. Another option might be to use bootstrap or jaccknife
    approaches to calibrate the gamma distribution.

- Suppose we have a collection of effect estimates
  $\hat{\theta}_1, \ldots, \hat{\theta}_m$ and their corresponding standard
  errors $s_1, \ldots, s_m$. Let
  $\tau^2 = {\rm Var}(\theta_1, \ldots, \theta_m)$ denote the variance of the
  true (unknown) effects, and let $\sigma^2 = (s_1^2 + \cdots + s_m^2)/m$
  denote the average sampling variance. According to the
  [law of total variation](https://en.wikipedia.org/wiki/Law_of_total_variance),
  the population variance of the $\hat{\theta}_j$ is $\sigma^2 + \tau^2$. The
  statistic $I^2 = \tau^2 / (\tau^2 + \sigma^2)$ is a measure of the
  proportion of total variance due to study heterogeneity.

- $I^2$ can be estimated as $\hat{I}^2 = 1 - (m-1)/Q$. This provides a
  [method of moments](<https://en.wikipedia.org/wiki/Method_of_moments_(statistics)>)
  estimate of $I^2$, whereas the estimates using mixed modeling software are
  typically based on maximum likelihood estimation.

- Although $I^2$ always lies between 0 and 1, $\hat{I}^2$ can be negative, and
  indeed when $I^2 = 0$ (no heterogeneity), $\hat{I}^2$ is negative around
  half of the time.

- Inference for variance fractions and
  [intra class correlations](https://en.wikipedia.org/wiki/Intraclass_correlation)
  are known to be a challenging problem. In a small meta-analysis (with few
  studies), the sample estimate $\hat{I}^2$ is biased and its confidence
  interval under-covers the true $I^2$, see
  [here](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-015-0024-z)
  for more details.

- The bias in $I^2$ depends on whether negative values of $\hat{I}^2$ are
  truncated to zero. If no truncation is performed, the bias when $I^2=0$ is
  negative, approximately $-2/(m-2)$. If truncation is performed, the bias is
  positive, and is around 0.15 for very small meta-analyses, and becomes
  smaller as the number of studies in the meta-analysis grows. These are
  worst-case biases at $I^2=0$. If heterogeneity is present, the bias is less.
  For example, if we have 50 studies and truncate at zero, the bias is
  approximately 0.06.

- An alternative to working with $I^2$ is to focus directly on $\tau^2$.
  Arguably this was favored by Cochran himself. A methof of moments estimate
  for $\tau^2$ is
  $\widehat{\rm var}(\hat{\theta}_1, \ldots, \hat{\theta}_m) - {\rm Avg}(\hat{s}_1, \ldots, \hat{s}_m)$.
  It is possible to put a confidence interval around this value using the
  method of jackknife empirical likelihood.

## Meta regression

- When study characteristics have been quantified in a consistent way across
  studies, it is possible to "partial out" the contributions of specific study
  characteristics to the overall heterogeneity.

- For example, suppose we have a meta-analysis that considers the effect of a
  primary treatment ($X=0,1$), with some studies excluding people with prior
  treatment (i.e. only enrolling newly-diagnosed subjects) while others do not
  make such an exclusion. Let $Z=0,1$ based on whether subjects with prior
  treatment are excluded ($Z=1$) or when no such restriction is imposed
  ($Z=0$). We can then fit a linear model
  $E[Y] = \beta_0 + \beta_1X + \beta_2Z$, where $Y$ is the reported treatment
  effect. This model should be fit with
  [generalized least squares](https://en.wikipedia.org/wiki/Generalized_least_squares)
  using inverse variance weights, with the variances being the squares of the
  reported standard error for each study effect.

- This framework can also be used to consider treatment effect modifiers.
  Continuing with the preceding example, we can fit the model
  $E[Y] = \beta_0 + \beta_1X + \beta_2Z + \beta_3XZ$. Then the treatment
  effect for populations excluding subjects with prior treatment is
  $\beta_1 + \beta_3$, while the treatment effect for populations that do not
  make this exclusion is $\beta_1$.

## Network meta-analysis

- In a basic two-arm study, the treatment effect is often estimated by taking
  the average response among treated subjects and subtracting from it the
  average response among control subjects. That is, the treatment effect is
  $D = \bar{Y}(X=1) - \bar{Y}(X=0)$. An alternative approach is to model the
  average responses for the various arms as separate observations. Suppose we
  have $K$ possible treatments, and let $Y_{ij}$ denote the average response
  of subjects in arm $j$ who were enrolled in study $i$.

- We can posit a mean-structure model $E[Y_{ij}] = \mu + \alpha_i + \beta_j$,
  where $\mu$ is an intercept, $\alpha_i$ is a study effect, and $\beta_j$ is
  an arm effect. If study or study/arm characteristics are known, let $Z_{ij}$
  be a vector of characteristics for arm $j$ in study $i$, and use a mean
  structure model such as
  $E[Y_{ij}] = \mu + \alpha_i + \beta_j + \gamma^\prime Z_{ij}$ The overall
  model would be $Y_{ij} = E[Y_{ij}] + s_{ij}\epsilon_{ij}$, where $s_{ij}$ is
  the reported standard error for $Y_{ij}$ and the $\epsilon_{ij}$ are
  independent unit-standard deviation random terms capturing unexplained
  variation.

- In many cases $\alpha_i$ will be modeled as a random effect, with variance
  $\tau_\alpha^2$, although fixed effects analysis is also an option.

- The variance for this model should be heteroscedastic, with
  ${\rm Var}[Y_{ij}]$ being equal to the reported standard error for arm $j$
  in study $i$. If sample sizes are reported but standard errors are not, it
  is possible to model the variance as ${\rm Var}[Y_{ij}] \propto 1/n_{ij}$,
  where $n_{ij}$ is the sample size.

- This type of approach can be used even when each study uses only a subset of
  the arms. When there are large number of treatments, we can visualize the
  data as a
  [graph](<https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)>), where
  treatments $j$ and $k$ are connected in the graph if these two treatments
  are ever present in the same trial. This opens up the possibility of making
  _indirect comparisons_ between pairs of treatments that have never been
  compared _directly_ in a single trial. This framework has been referred to
  as a _network meta analysis_.

## Biases in study reporting

- A high-quality meta-analysis should assess the extent to which the selection
  of source materials may introduce bias.

- One potential source of selection bias in meta-analysis at the level of the
  studies (not the subjects) is
  [publication bias](https://en.wikipedia.org/wiki/Publication_bias). This
  refers to the possibility that inconclusive studies or studies that
  contradict the dominant narrative are less likely to be published than
  studies that support the consensus point of view.

- In recent years, efforts have been made to encourage _pre-registration_ of
  both interventional and observational studies. With pre-registration, it
  becomes possible to assess the study characteristics of all registered
  studies conducted to address a specific question, regardless of whether they
  have been published.

- If all studies are assessing the same effect, then the scatter of the
  standard error of the estimated treatment effect (on the vertical axis)
  against the estimated treatment effect (on the horizontal axis) can be used
  to assess publication bias. This is called a
  [funnel plot](https://en.wikipedia.org/wiki/Funnel_plot).

- Some funnel plots use the precision as the vertical coordinate rather than
  the standard error (the precision is the reciprocal of the standard error).

- The logic behind a funnel plot is that any given quantile of the effect
  estimates should scale linearly with the standard error. Thus, the effect
  estimates should be distributed in a cone with vertex at the origin.

- Publication bias may be reflected in an empty region in the funnel plot,
  usually corresponding to studies with higher standard error that contradict
  the hoped-for narrative.

- Funnel plots can be interpreted visually, or a test such as _Egger's test_
  can be used. Funnel plots can be useful but have been criticized for having
  low power. Moreover, funnel plots can be misleading if there are systematic
  differences in the effect being estimated between larger and smaller
  studies.

- The
  [trim and fill method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6571372)
  can be used to impute missing studies in a funnel plot, to compensate for
  publication biases.

## Case studies and other links

[Misunderstandings about Q and Cochran's Q test in meta-analysis](https://onlinelibrary.wiley.com/doi/epdf/10.1002/sim.6632)

[On the moments of Cochran's Q statistic under the null hypothesis, with application to the meta-analysis of risk difference](https://onlinelibrary.wiley.com/doi/10.1002/jrsm.54)

[Bootstrap procedures for testing homogeneity hypotheses](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=e4a7d2860fcf4f5e211412a4f7e32b9fb6e2b973)

[GFR in Healthy Aging: an Individual Participant Data Meta-Analysis of Iohexol Clearance in European Population-Based Cohorts](https://pmc.ncbi.nlm.nih.gov/articles/PMC7350990)

[A meta-analysis of GFR slope as a surrogate endpoint for kidney failure](https://pubmed.ncbi.nlm.nih.gov/37330614/)

[metafor R package](https://cran.r-project.org/web/packages/metafor/index.html)
