# Rigorous statistical design

- What is the goal of conducting
  [research](https://en.wikipedia.org/wiki/Research)?

  - To gain insight into the the mechanisms underlying the natural world.

  - To describe and characterize the state of the natural world.

  - To produce new scientific knowledge.

  - To test theories about mechanisms that explain the natural world.

  - To produce new concepts, methods, and theories.

- Everything else we discuss in this workshop is subservient to the goals stated
  above.

- How (at a very high level) should we go about the process of conducting
  research?

- There is much deep and fundamental work in the
  [philosophy of science](https://en.wikipedia.org/wiki/Philosophy_of_science).
  We will only cover this topic briefly and superficially here, in favor of more
  pragmatic topics.

  - For those interested, here is a readable account of the work of the very
    influential philosopher of science
    [Karl Popper](https://iep.utm.edu/pop-sci/#:~:text=Popper's%20deductive%20account%20of%20theory,are%20likely%20to%20be%20true).

- At a very high level, most research is either
  [inductive](https://en.wikipedia.org/wiki/Inductive_reasoning) or
  [deductive](https://en.wikipedia.org/wiki/Deductive_reasoning). Inductive
  research is data-driven and empirical, that is our focus here. Deductive
  research emphasizes starting with "first principles", and seeing what can be
  derived logically from them.

  - Many researchers combine data-driven and first principles approaches in
    their work.

- Empirical research generally involves design (planning), data collection, and
  data analysis.

  - Design refers to the process of developing a rationale and goals for the
    research, stating (falsifiable)
    [hypotheses](https://en.wikipedia.org/wiki/Hypothesis), and documenting how
    the data will be collected, analyzed, and interpreted.

  - Data collection refers to everything involved with making and recording
    measurements.

  - Data analysis refers to the executation of a data analysis plan, which
    includes both carrying out the analysis and interpreting the results in
    relation to the (ideally) pre-specified hypotheses.

- There are many different types of research "studies". We are not aiming here
  to produce a typology for all possible situations. Here are some of the major
  features that differentiate among designs of research studies.

  - In an _interventional study_, the units are manipulated by the researcher.
    This is also sometimes called an
    [experiment](https://en.wikipedia.org/wiki/Experiment).

  - In an
    [observational study](https://en.wikipedia.org/wiki/Observational_study),
    the experimental units are not manipulated by the researcher.

  - A _controlled_ study involves a comparison to a group (arm) that is
    untreated, or that is treated with a conventional well-understood treatment
    (e.g. "standard of care" in clinical research).

  - A
    [randomized controlled study](https://en.wikipedia.org/wiki/Randomized_controlled_trial)
    (or "trial", RCT) is a controlled study in which the subjects are assigned
    to the treatment or control group by randomization.

  - In a [longitudinal study](https://en.wikipedia.org/wiki/Longitudinal_study),
    each unit is observed on multiple occasions. This is also known as a panel
    study.

  - In a
    [cross-sectional study](https://en.wikipedia.org/wiki/Cross-sectional_study),
    each unit is observed on a single occasion.

  - In a [cohort study](https://en.wikipedia.org/wiki/Cohort_study), a group of
    units are followed over time. The units are initially either similar, or
    differ primarily with respect to a specific factor of interest. Exposures
    happpen naturalistically and we can assess at the end of the study which
    exposures occurred and what outcomes followed the occurence of these
    exposures.

    - A
      [prospective cohort study](https://en.wikipedia.org/wiki/Prospective_cohort_study)
      involves collecting data as the events of interest occur in real time.

    - A
      [retrospective cohort study](https://en.wikipedia.org/wiki/Retrospective_cohort_study)
      involves data that were collected for purposes other than your research
      study (i.e. for other research studies or for administrative purposes).

  - In a [case/control study](c) units are deliberately selected at the outset
    to belong to contrasting states of a factor of interest.

  - A _micro-longitudinal_ study is a study with relatively few unit, but each
    unit is measured intensively on multiple characteristics over time.

  - An
    [adaptive interventional study](<https://en.wikipedia.org/wiki/Adaptive_design_(medicine)>)
    is one in which the treatments assigned to units depends on the results of
    previously collected data within the same study, based either on an _interim
    analysis_ or on _continual reassessment_ of findings. In some cases the
    design updates only impact patients not yet enrolled, while in other cases
    the design updates impact the people currently undergoing treatment (e.g. in
    a [SMART](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4167891) design).

## Measurement

- Some quantities can be directly
  [measured](https://en.wikipedia.org/wiki/Measurement) in a more or less exact
  manner, like a person's height.

- Some quantities are difficult to measure because they change rapidly (heart
  rate) or differ based on who is doing the measurement or how the measurement
  is taken (blood pressure).

- Some quantities are difficult to measure because they place a lot of burden on
  the subjects (invasive diagnostic tests).

- Some measurements require human subjects to report behaviors, for example food
  intake, sleep, or substance use. These are burdensome and notoriously
  inaccurate. They are subject to
  [recall bias](https://en.wikipedia.org/wiki/Recall_bias) and
  [social desirability bias](https://en.wikipedia.org/wiki/Social-desirability_bias),
  among other issues.

- Technology is rapidly tranforming the landscape for measurements of human
  behaviors (wearables for activity and sleep, real time monitors for e.g. blood
  glucose, bio-assays e.g. for nicotine use).

- Some quantities are difficult to measure because they are abstract and there
  is debate about exactly what the quantity means. The term "construct" is used
  in many cases to refer to things like social attitudes, or skills (language or
  math skills) that cannot be simply and directly measured.

  - There is a branch of
    [psychometric theory](https://en.wikipedia.org/wiki/Psychometrics) that
    deals with methods for forming constructs out of item-level (scalar)
    variables.

  - Statistical methods such as
    [factor analysis](https://en.wikipedia.org/wiki/Factor_analysis) can be used
    to produce _scales_ from a collection of items, e.g. depression, overall
    physical health, hopefullness.

  - The [Delphi method](https://en.wikipedia.org/wiki/Delphi_method) is used to
    objectively produce items that can be used to construct novel scales.

- Most values that are measured are measured with some level of error.
  Measurement errors can be classified into two broad types.

  - Systematic errors are a type of bias in the data. These are errors that
    would occur persistently in one direction upon repeated sampling. In the
    context of measuring blood pressure, there is a well-known systematic error
    known as "white coat hypertension" in which some peoples' blood pressure
    rises due to the stress of interacting with the person doing the
    measurement.

  - Random errors are a type of error that is transient or specific to one
    instance of making a measurement. Using blood pressure again as an example,
    there are substantial random errors due to the perception of detecting a
    pulse as the blood pressure cuff is depressurized. The variance of these
    random errors may depend on the skill of the person doing the measurement.

  - Random errors can futher be classifies as classical errors and
    [Berkson errors](https://en.wikipedia.org/wiki/Berkson_error_model).

    - Classical errors arise due to additive random measurement error that is
      independent of the true value. $X_{\rm obs} = X_{\rm true} + \eta$, where
      $E[\eta | X_{\rm true}] = 0$, where $\eta$ is the random measurement
      error.

    - Berkson errors arise when $E[\eta | X_{\rm obs}] = 0$. A typical example
      is where we wish to study the yield of a chemical reaction at a specific
      temperature. We set the temperature to a desired level $T$ and use a
      heating device to produce the desired temperature. Since the heating
      device can never be perfect, the actual temperature will arguably follow
      $X_{\rm true} = X_{\rm obs} + \eta$, where $\eta$ is measurement error
      satisfying $EE[\eta | X_{\rm obs}] = 0$.

## Causality and confounding

- Most research aims to understand relationships among factors. In health
  research we often speak of _exposures_ and _outcomes_, or _treatments_ and
  outcomes. The term "exposure" is typically used when the factor occurs
  naturalistically whereas "treatment" is used for interventions or experimental
  manipulations.

- If $X$ is an exposure and $Y$ is an outcome, there is an _association_ between
  $X$ and $Y$ if $X$ and $Y$ are not statistically independent. A _causal_
  association is one in which manipulation of $X$ would lead to a change in $Y$
  (on average or in some statistical sense). Below are some examples of
  associations that are unlikely to be causal associations:

  - More ice cream is consumed on days in which the number of drowning deaths is
    greater. If we were to ban the consumption of ice cream, would we see a
    reduction in drowning deaths?

  - From 2000-2020, the marriage rate was lower in years when internet usage was
    greater. If the internet had not been invented, would marriage rates have
    followed a different pattern?

- Why are some associations not causal?

  - The main explanation for associations being non-causal is the role of a
    _confounding variable_ (or "confounder"). This is a variable that causally
    influences both the exposure and the outcome. Let's reconsider the two
    examples above:

    - Temperature and rainfall are likely confounding factors in the
      relationship between ice cream consumption and drowning. On warm days with
      little rain, people enjoy eating ice cream and these are the same
      conditions that might lead people to go to the pool or the beach. When
      there are more people at the pool or beach, there is a greater risk for
      drowning deaths occurring.

    - [Secular trends](https://en.wikipedia.org/wiki/Secular_variation) relating
      to technological and social change are very likely acting as confounders
      in the relationship between internet usage and marriage rates.

  - A confounder can be either known or unknown, and measured or unmeasured.

  - Unfortunately there is no automatic process for identifying confounders.

  - Randomization is the best way to limit or eliminate the risk of confounding.
    If the treatment is assigned at random, it is impossible for it to be
    causally influenced by any factor (regardless of whether the factor is known
    or unknown, measured or unmeasured).

  - If randomization is impossible, the next best approach is to achieve
    _balance_ of all measured confounders between the exposure arms.

## Randomization

- In research in which units are assigned to treatment arms, assigning the units
  [randomly](https://en.wikipedia.org/wiki/Randomization) guarantees that there
  is no confounding at the population level. That is, there can be no
  statistical dependence between the assigned treatments and characteristics of
  the units.

- _Balance_ refers to the extent to which covariates have equal distributions
  across treatment arms. Confounding will generally lead to a lack of balance,
  but even if there is no confounding, there can be a lack of balance,
  especially when the sample size is small. The is because randomization (which
  guarantees no confounding) produces balance on average, but in any given study
  randomization will usually not produce perfect balance.

- Bias can result from lack of balance, even in the absence of counfounding.

- For measured (potential) confounders, we can quantify and compensate for any
  lack of balance. For unmeasured or unknown confounders, randomization is
  usually the only practical way to avoid bias due to confounding.

- The most basic type of randomization is simple randomization, where each unit
  is independently assigned to a treatment group, either with uniform
  probabilities (equal porobability of assignment to each arm) or with
  probabilities that are pre-determined to achieve desired relative group sizes
  (e.g. assigning to treatment with twice the probability of assigning to
  control).

- Numerous approaches to randomization have been devised that aim to preserve
  the benefits of randomization, while also producing balance with respect to
  observed confounders. Some of these approaches are:

  - [Stratified randomization](https://en.wikipedia.org/wiki/Stratified_randomization)

  - Minimization

- A concern in studies where units are recruited sequentially is bias on the
  part of the research team in deciding how to interpret eligibility criteria.
  If the research team knows that the next subject to be recruited will be
  assigned the treatment, and the next subject appears to be someone who is less
  likely to do well on the treatment, the research team may elect to interpret
  the eligibility requirements strictly so that this subject is not recruited
  into the study. Conversely, if it is knownm that the next subject to be
  assigned will be in the control arm, there may be a bias to include subjects
  who are sicker or less likely to benefit for some reason. Maintaining some
  degree of randomness in the assignments helps to mitigate this issue.

## Foundations of statistical inference

- Most statistical data analysis is based on probability modeling. That is, we
  posit that our data are a random sample from a probability distribution
  $P_\theta$, where $\theta$ is a
  [parameter](https://en.wikipedia.org/wiki/Statistical_parameter) that captures
  aspects of the scientific research question.

- Probability theory considers the properties of a sample $D$ from a given
  probability model $P$. Statistical inference is the reverse of this -- given a
  random sample $D$, what can we say about the probability distribution $P$?

- Statistical inference generally begins with
  [estimation](https://en.wikipedia.org/wiki/Estimation). Formally, this inolves
  devising a function $\hat{\theta}(D)$, where $D$ is the _observed data_, such
  that $\hat{\theta}(D)$ is likely to be close to the true parameter value
  $\theta$.

- It is common to refer to $P_\theta$ as the
  [population](https://en.wikipedia.org/wiki/Statistical_population), $D$ as the
  [sample](<https://en.wikipedia.org/wiki/Sampling_(statistics)>), and
  $\hat{\theta}$ as the _parameter estimate_.

- There are many ways to obtain parameter estimates, two of the most common are
  the
  [method of moments](<https://en.wikipedia.org/wiki/Method_of_moments_(statistics)>),
  and [maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood)
  analysis.

- Statistical estimators may exhibit desirable properties, including
  [unbiasedness](<https://en.wikipedia.org/wiki/Bias_(statistics)>),
  [consistency](<https://en.wikipedia.org/wiki/Consistency_(statistics)>), and
  [efficiency](<https://en.wikipedia.org/wiki/Efficiency_(statistics)>).

- The [standard error](https://en.wikipedia.org/wiki/Standard_error) is a key
  tool for characterizing the precision or uncertainty in a parameter estimate.
  It is the standard deviation of the
  [sampling distribution](https://en.wikipedia.org/wiki/Sampling_distribution)
  of the random variable $\hat{\theta}$, which is induced by the underlying
  distribution of the data, $P(D)$.

- If the sampling distribution of $\hat{\theta}$ is approximately
  [Gaussian](https://en.wikipedia.org/wiki/Normal_distribution), then the
  standard error is all one needs to fully characterize the estimation errors of
  an unbiased estimator.

- Many research questions cannot be reduced to a single scalar parameter, so the
  parameter is often a vector. Vector-valued parameters can often be partitioned
  into parameters of primary interest and
  [nuisance parameters](https://en.wikipedia.org/wiki/Nuisance_parameter).

## Statistical power

## Experimental design

## Surveys

- A _sample survey_ is a research tool in which the goal is to quantify the
  state of a population, with a primary focus on achieving low bias for a
  defined target population.

- Unlike most trials and studies (cohort, case/control, etc.) the goal of a
  survey is not usually to assess the effect of an exposure or intervention, and
  a survey does not usually have "arms" corresponding to different treatments or
  exposures.

- Most sample surveys use random selection to identify subjects. The resulting
  sample is called a _probability sample_.

- A special type of survey is a [census](https://en.wikipedia.org/wiki/Census),
  which aims to measure the entire population, rather than measureing only a
  sample of a population.

  - In most cases a census is either impossible or impractical. A well-conducted
    sample survey can give results that are for all practical purposes just as
    informative as a census, usually at a much lower expense.

## Sampling

- In surveys, as well as in some other contexts, it is important to carefully
  sample units from a population in such a way that unbiased results can be
  obtained from the sample.

- The most basic type of sampling is a _simple random sample_ (SRS), which is a
  sample of size $k$ from a population of size $n$ in which any subset of size
  $k$ is equally likely to be selected.

- An SRS can be obtained if we have a _sampling frame_ which is a complete list
  of all units in the population. For example, if we want to sample thje
  employees of a company or the students at a school, a sampling frame would
  generally be available and it would be practical to obtain a simple random
  sample from it.

  - For larger or more intangible populations, like the total population of a
    geographic region, a sampling frame is usually not available. Even if a
    sampling frame is available, it may be difficult to reach certain units, or
    (in the case of human subjects) people may decline to participate.

- A common type of survey is a _cluster sample_. In a cluster sample, the
  population is partitioned into many _primary sampling units_ (PSU), which
  often are geographic areas. Then, a limited number of PSUs are selected at
  random (possibly with probabilities proportional to size), and units are
  selected from the PSUs using simple random sampling (or something that
  approximates it). In some cases there are two or more levels, e.g. randomly
  select cities, then randomly select schools within cities, then randomly
  select students within schools.

- In a _stratified sample_, the population is partitioned into groups, e.g.
  people are stratified by race, and separate samples are drawn for each
  stratum. In this way, the sample size per stratum is fixed rather than random,
  which woujld be the case if a simple random sample were obtained. A rationale
  for stratification is to guarantee coverage of people from all groups of
  interest, including groups that are relative small.

- Weighting

## Missing data and selection bias

- Loss to follow-up

- Types of missing data

  - Missing completely at random (MCAR)

  - Missing at random (MAR)

  - Missing not at random (MNAR)

- Collider stratification bias

## Methods for observational data

### Difference in difference

### Regression adjustment

### Regression discontinuity

### Natural experiments

### Matching

### Stratification

### Propensity scores

### Synthetic controls

## Methods for designed experiments
