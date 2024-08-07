# Foundations of data science

## Probability

### Basic concepts of probability

- How do we formally study outcomes that occur "at random"?

- The most basic example is a random outcome that can have only two states, e.g.
  flipping a coin. We can talk about the "probability of observing a head" and
  the "probability of observing a tail". This is called _discrete probability_.

- We often wish to use probability in situations where the outcome is
  _quantitative_ (also known as _numeric_ or _continuous_). If I scoop a volume
  of water from the ocean, what is the probability that I scoop exactly 300ml of
  water? Arguably it is zero. Actually, every possible outcome has probability
  zero, yet some outcome always occurs.

- With quantitative outcomes, we should only assign probabilities to ranges, not
  to specific outcomes, e.g. what is the probability that I scoop between 300ml
  and 310ml of water?

- It's not straightforward to use math to precisely talk about randomness.
  Mathematical probability only became rigorous in the 20th century, largely due
  to the formal development of
  [measure theory](<https://en.wikipedia.org/wiki/Measure_(mathematics)>) and
  the work of [Kolmogorov](https://en.wikipedia.org/wiki/Andrey_Kolmogorov).

- Key terms to understand:

  - [Sample space](https://en.wikipedia.org/wiki/Sample_space): all outcomes
    that can be observed

  - [Event](<https://en.wikipedia.org/wiki/Event_(probability_theory)>): a
    subset of the sample space to which we can assign a probability. Not all
    subsets are events.

  - [Probability measure](https://en.wikipedia.org/wiki/Probability_measure): a
    function that assigns a probability to each event.

- A probability measure has these properties:

  - The probability of every event is non-negative.

  - The probability of the event that includes the entire sample space is equal
    to 1.

  - If $E$ is an event and $E^c$ is its complement, then the probability of
    $E^c$ is 1 minus the probability of $E$.

  - If events $A$ and $B$ are disjoint (they contain no elements in common),
    then the probability of the union of $A$ and $B$ is the sum of the
    probability of $A$ and the probability of $B$.

- A [random variable](https://en.wikipedia.org/wiki/Random_variable) is a symbol
  such as $X$ that represents a value drawn from a probability distribution. We
  can write expressions such as $P(X \le 3)$, $P(X = 4)$, $P(1 \le X < 2)$, etc.
  Keep in mind that $X$ has no fixed value, every time we view or refer to $X$
  its value changes.

### Probability distributions

- There are several effective ways to represent a probability distribution.

  - If the sample space is finite, we can list the points in the sample space
    and their probabilities. This type of probability distribution takes the
    form of a table, which is often called a
    [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function)
    (pmf).

  - If the sample space is countable, individual "atoms" (points in the sample
    space) have positive probability and there is a probability mass function
    mapping each atom to its probability.

  - If the sample space is the real line, or an interval on the real line, we
    have the
    [cumulative distribution function (CDF)](https://en.wikipedia.org/wiki/Cumulative_distribution_function)
    $F(t) = P(X \le t)$. The probabilities of all events can be inferred from
    the CDF, e.g. the probability that $X$ is between $1$ and $2$ or between $6$
    and $7$ is $F(2) - F(1) + F(7) - F(6)$.

  - If there is an atom with positive mass, the CDF is discontinuous at that
    point, e.g. $P(X=x)$ is the jump in the CDF at $x$.

  - Many common distributions whose sample space is the real numbers have a
    [density](https://en.wikipedia.org/wiki/Probability_density_function), which
    is a function $f$ such that $\int_a^bf(x)dx = F(b) - F(a)$ is the
    probability that $X$ falls between $a$ and $b$.

  - Some other ways to represent a probability distribution are through its
    [quantile function](https://en.wikipedia.org/wiki/Quantile_function) and its
    [characteristic function](<https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)>).

- To better understand the challenges of representing a probability
  distribution, we need to distinguish between different types of
  [infinity](https://en.wikipedia.org/wiki/Infinity).

  - [Countable infinity](https://en.wikipedia.org/wiki/Countable_set) refers to
    a set such as the integers (the definition of _countable_ is that the set is
    in one-to-one correspondence with the integers). It is possible to have a
    mass function on a countably infinite set since a
    [series](<https://en.wikipedia.org/wiki/Series_(mathematics)>) can be
    summable. For example, $(1 - r)(1 + r + r^2 + r^3 + \cdots) = 1$ for
    $0 < r < 1$, so we can create a PMF on the countably infinite sample space
    $0, 1, 2, \ldots$ as $P(X=j) = (1-r)r^j$.

  - [Uncountable infinity](https://en.wikipedia.org/wiki/Uncountable_set) is a
    more challenging concept. For our purposes we can focus on the
    [real numbers](https://en.wikipedia.org/wiki/Real_number). The real numbers
    are not countable but we frequently wish to work with them in applied
    analysis. When working with the real numbers, we must contend with the
    unavoidable fact that there will always be
    [non-measurable sets](https://en.wikipedia.org/wiki/Non-measurable_set),
    i.e. subsets ${\cal S} \subset {\mathbb R}$ to which it is impossible to
    assign a probability.

### Examples of probability distributions

- The uniform distribution is either the
  [distribution](https://en.wikipedia.org/wiki/Discrete_uniform_distribution) in
  which every atom has the same probability (for finite sample spaces), or (for
  distributions on a real interval (a, b)), it is a
  [distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
  in which the probability of any interval is proportional to its length.

- The
  [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution)
  with CDF $P(X \le t) = 1 - \exp(-\lambda t)$ and density
  $f(x) = \lambda
  \exp(-\lambda x)$. The value of $\lambda$ is a
  [parameter](https://en.wikipedia.org/wiki/Statistical_parameter), so there are
  actually infinitely many exponential distributions, determined by the value of
  $\lambda$.

- The
  [normal (Gaussian) distribution](https://en.wikipedia.org/wiki/Normal_distribution)
  with density $f(x) = \exp(-(x-\mu)^2/2\sigma^2)/\sqrt{2\pi\sigma^2}$. The
  values of $\mu$ and $\sigma$ are parameters, referring to the expected value
  and standard deviation, respectively.

### Measures of location and dispersion

- Probability distributions are somewhat unwieldy to work with, it is helpful to
  summarize them in a few informative numbers.

- _Measures of [location](https://en.wikipedia.org/wiki/Central_tendency)_ aim
  to capture the central value of a distribution. The mean and median (discussed
  in more detail below) are measures of location, and there are many more.

- _Measures of
  [dispersion](https://en.wikipedia.org/wiki/Statistical_dispersion)_ aim to
  capture how spread out or heterogeneous are the values of the distribution.
  The standard deviation and interquartile range (discussed in more detail
  below) are measures of dispersion, and there are many more.

### Expectation

- The [expectation](https://en.wikipedia.org/wiki/Expected_value) is a summary
  statistic of a quantitative random variable $X$.

- For a random variable with a density $f(x)$, the expectation $E[X]$ is
  $\int xf(x)dx$. For a random variable with a mass function the expectation is
  $E[X] = \sum xf(x)$, where $x$ ranges over the domain of $X$.

- The expectation solves the optimization problem
  ${\rm argmin}_\theta E[(X-\theta)^2]$

- The expectation can be used to define the _center_ of a distribution, and has
  a physical analogy in being the balancing point if the mass is arranged along
  a beam.

- The expectation may not exist, and it may be infinite.

- Expectations are linear. If $X$ and $Y$ are random variables, and $c$ is a
  constant, then $E[X + cY] = E[X] + cE[Y]$.

- The terms _mean_ and _average_ may be used synonymously with "expectation."

### Quantiles and moments

Most summary statistics have one of two mathematical forms, as either a
[moment](<https://en.wikipedia.org/wiki/Moment_(mathematics)>) or as a
[quantile](https://en.wikipedia.org/wiki/Quantile).

- Moments are expectations of (possibly) transformed random values.

  - The expected value $E[X]$ is a moment, but so is $E[X^4]$ and $E[\exp(X)]$.

  - The $k^{\rm th}$ raw moment is $E[X^k]$ and the $k^{\rm th}$ central moment
    is $E[(X - EX)^k]$.

  - The [variance](https://en.wikipedia.org/wiki/Variance) is the second central
    moment $E[(X - EX)^2]$. It is a measure of dispersion but its units are the
    square of the units of $X$. Therefore we often use the _standard deviation_
    which is the square root of the variance. It has the same units as the data.

  - The $k^{\rm th}$ standardized moment is the $k^{\rm th}$ raw moment of the
    standardized variate $(X - EX)/{\rm SD}(X)$

  - The third standardized moment $E[Z^3]$ where $Z = (X - EX)/{\rm SD}(X)$ is a
    measure of [skewness](https://en.wikipedia.org/wiki/Skewness). The fourth
    standardized moment measures
    [kurtosis](https://en.wikipedia.org/wiki/Kurtosis), higher order moments are
    subtle to interpret.

- Quantiles are based on the inverse CDF. If $F(t) = P(X \le t)$ is the CDF then
  the quantile function is $Q(p) \equiv F^{-1}(p)$. If $F$ is not invertible
  then $Q(p)$ is defined as ${\rm inf}\\{t \\;|\\; F(t) \ge p\\}$. The
  $p^{\rm th}$ quantile answers the question "for a given $p$, what is the value
  $t$ such that $p$ of the mass of the distribution falls on or below $t$".

  - The median is a measure of location that is a quantile, it is $Q(1/2)$.

  - Quantiles can be used to define measures of dispersion, e.g. the
    [interquartile range](https://en.wikipedia.org/wiki/Interquartile_range)
    which is $Q(3/4) - Q(1/4)$ or the
    [median absolute deviation (MAD)](https://en.wikipedia.org/wiki/Median_absolute_deviation)
    which is the median of $|X - EX|$.

  - Quantile analogues of higher order moments exist, e.g. a quantile-based
    measure of skewness is

    $(Q(3/4) - 2Q(1/2) + Q(1/4)) / (Q(3/4) - Q(1/4)).$

    The theory of [L-moments](https://en.wikipedia.org/wiki/L-moment) provides a
    general means for constructing higher order summary statistics from
    quantiles.

### Conditional distributions, expectations and variances

- In most research we are working with more than one quantity at a time. This
  leads to the concept of a
  [random vector](https://en.wikipedia.org/wiki/Multivariate_random_variable)
  that has a
  [joint distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution).
  Suppose that we have two jointly-distributed random variables $X$ and $Y$,
  yielding a random vector $[X, Y]$.

  - The joint CDF of $X$ and $Y$ is the function
    $F(s, t) = P(X \le s, Y \le t)$. There is also a concept of a joint density
    (which does not always exist). But generalizing quantiles to joint
    distributions is not straightforward.

  - If $f(x, y)$ is the joint density of $(X, Y)$, then the marginal density of
    $X$ is $f(x) = \int f(x, y)dy$, and an analogous result gives the marginal
    density of $Y$.

  - The conditional distribution $P(Y | X)$ is the distribution of $Y$ that
    results when the value of $X$ is known. Suppose that there is a joint
    density $f(x, y)$. Then the density of the conditional distribution
    $P(Y |
    X)$ is $f(x, y) / f(x)$.

- [Conditional expectation](https://en.wikipedia.org/wiki/Conditional_expectation)

  - Suppose that $Y$ is quantitative and is jointly distributed with $X$ (which
    may or may not be quantitative). The _conditional expectation_ of $Y$ given
    $X$, $E[Y|X=x]$ is (roughly speaking) the expected value of $Y$ when we know
    that $X$ takes on the value $x$. The rigorous definition of conditional
    expectation is somewhat different but this is the way that most people think
    about conditional expectation intuitively.

  - A special case is when $X$ has a finite sample space, so that it effectively
    partitions the population into groups. In this case, $E[Y | X=x]$ is the
    expected value of $Y$ when we are in group $x$.

  - The _double expectation theorem_ or _smoothing theorem_ or
    [law of total expectation](https://en.wikipedia.org/wiki/Law_of_total_expectation)
    states that $E[E[Y|X]] = E[Y]$. That is, if we compute the conditional
    expectation of $Y$ at each fixed value of $X$, and then average these values
    over the marginal distribution of $X$, we get the same result as if we
    compute the marginal mean of $Y$ directly.

- Conditional variance

  - The conditional variance is analogous to the conditional expectation.
    ${\rm Var}[Y|X=x]$ is the variance restricted to $(X, Y)$ values such that
    $X=x$.

  - The
    [law of total variation](https://en.wikipedia.org/wiki/Law_of_total_variance)
    states that ${\rm Var}(Y) = {\rm Var}[E[Y|X]] + E[{\rm Var}[Y|X]]$

  - The term ${\rm Var}[E[Y|X]]$ is the _between variation_ while the term
    $E[{\rm Var}[Y|X]]$ is the _within variation_. The law of total variation
    states that the overall variation is the sum of the between and within
    variations.

  - The identity can also be written as:

    $1 = {\rm Var}[E[Y|X]] / {\rm Var}(Y) + E[{\rm Var}[Y|X]] / {\rm
    Var}(Y)$

    This shows that the proportion of the variance in $Y$ explained by $X$,
    ${\rm Var}[E[Y|X]] / {\rm Var}(Y)$ is complementary to the proportion of
    variance in $Y$ that is not explained by $X$,
    $E[{\rm Var}[Y|X]] / {\rm Var}(Y)$.

  - In regression analysis, ${\rm Var}[E[Y|X]] / {\rm Var}(Y)$ is known as the
    [coefficient of determination](https://en.wikipedia.org/wiki/Law_of_total_variance),
    the proportion of explained variance, or the $R^2$.

### Independence and measures of association

- Two jointly distributed random variables $X$ and $Y$ are
  [independent](<https://en.wikipedia.org/wiki/Independence_(probability_theory)>)
  if

  $P(X \in E_1 \\; \\& \\; Y \in E_2) = P(X \in E_1) \cdot P(Y \in E_2)$

  for all events $E_1$ and $E_2$. This essentially means that knowing the value
  of $X$ tells you nothing about the value of $Y$.

- If $X$ and $Y$ are independent then $E[Y|X=x] = E[Y]$ for all values of $x$,
  and ${\rm Var}[Y|X=x] = {\rm Var}[Y]$ for all values of $x$.

- The [covariance](https://en.wikipedia.org/wiki/Covariance) is a measure of the
  relationship between $X$ and $Y$. It is a moment that is defined to be
  $E[(X-EX)\cdot (Y-EY)]$. The covariance has some important properties:

  - The covariance of a random variable with itself is the variance:
    ${\rm Cov}(X, X) = {\rm Var}(X)$.

  - The covariance is symmetric: ${\rm Cov}(X, Y) = {\rm Cov}(Y, X)$

  - The covariance is [bilinear](https://en.wikipedia.org/wiki/Bilinear_form).
    This means that ${\rm Cov}(X+Y, Z) = {\rm Cov}(X, Z) + {\rm Cov}(Y, Z)$.

- The _standardized_ or _Z-scored_ version of a random variable is the variable
  $(X - E[X]) / {\rm SD}(X)$.

- The _correlation coefficient_ is the covariance calculated for standardized
  versions of $X$ and $Y$, that is
  $\rho \equiv E[(X-EX)\cdot (Y-EY)]/({\rm SD}(X)\cdot {\rm SD}(Y))$.

- The correlation coefficient always lies between $-1$ and $1$. When the
  correlation coefficient is equal to $1$, $Y$ is a linear function of $X$ with
  positive slope. When the correlation coefficient is equal to $-1$ $Y$ is a
  linear function of $X$ with negative slope. If the correlation coefficient is
  equal to zero then $X$ and $Y$ are said to be _uncorrelated_. The correlation
  coefficient is undefined if either ${\rm SD}(X) = 0$ or ${\rm SD}(Y) = 0$.

- Two independent random variables are necessarily uncorrelated. But the
  converse is not true. Two random variables can be uncorrelated, but not be
  independent.

- The
  [correlation coefficient](https://en.wikipedia.org/wiki/Correlation_coefficient)
  (often called the _Pearson_ or _product moment_ correlation coefficient) is a
  _measure of association_. Note that the product $(X-EX)\cdot(Y-EY)$ is
  positive when $X$ and $Y$ lie on the same side of their respective expected
  values, and is greater when they both lie far on the same side of their
  expected values. Thus, the correlation coefficient tends to be positive and
  larger when this happens frequently.

- The (Pearson) correlation coefficient is often said to be a measure of the
  _linear_ association between $X$ and $Y$, and strictly speaking this is true.
  [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet) shows
  many different joint distributions that have the same linear correlation and
  hence the same correlation coefficient. However the correlation coefficient is
  able to detect many forms of dependence beyond that which is strictly linear.

- There are many other measures of association besides the product moment
  correlation.
  [Spearman's correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
  is simply the product moment correlation calculated using the data ranks
  instead of their actual values. The
  [tau correlation](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)
  focuses on concordant orderings of pairs of values.

- The
  [distance correlation](https://en.wikipedia.org/wiki/Distance_correlationhttps://en.wikipedia.org/wiki/Distance_correlation)
  is a relatively recent innovation that yields a statistic that is equal to
  zero if and only if two random variables are independent. That is, it is a
  measure of any type of dependence, not only linear dependence.

- In many cases we wish to characterize the associations among several (more
  than two) random variables. One way to do this is using the
  [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix). If we
  have $d$ random variables, then the covariance matrix is a $d\times d$ matrix
  $C$ such that $C_{ij}$ is the covariance between the $i^{\rm th}$ and
  $j^{\rm th}$ of the random variables. Note that this implies that $C_{jj}$ (a
  diagonal element of $C$) is the variance of the $j^{\rm th}$ random variable.
  Covariance matrices have some special properties as matrices, which we will
  explore below in our review of linear algebra.

### Limits and concentration

- Some of the most powerful tools in probability are theoretical results that
  provide an understanding of how a complex distribution can be approximated by
  a simpler distribution, or how a sequence of more complex distributions may
  converge in some limiting sense to a simpler distribution.

- To facilitate discussion of limiting distributions, we will first introduce
  the notion of a sequence of random variables. A sequence of random variables
  of length $n$, denoted $X_1, X_2, \ldots, X_n$, may belong to one or more of
  the following important classes.

  - An
    [independent and identically distributed (IID) sequence](https://en.wikipedia.org/wiki/independent_and_identically_distributed_random_variables)
    has the property that the $X_i$ are mutually independent, and each has the
    same marginal probability distribution. Being IID implies that the joint
    distribution factors as $P(X_1, \ldots, X_n) = \prod_i P(X_i)$, and that
    $P(X_i) = P(X_j)$ for all $1 \le i, j \le n$.

  - A [stationary sequence](https://en.wikipedia.org/wiki/Stationary_sequence)
    is _shift invariant_ in the sense that
    $P(X_1, \ldots, X_m) = P(X_{j+1}, \ldots, X_{j+m})$ for all $j, m$.

  - An
    [exchangeable sequence](https://en.wikipedia.org/wiki/Exchangeable_random_variables)
    has the property that the joint distribution of any subset of a given size
    is the same. That is, if $I, J \subset 1, 2, \ldots$ with $|I|=|J|$, then
    $P(X[I]) = P(X[J])$.

  - An _m-dependent sequence_ is one in which subsets of the sequence are
    independent as long as no two values are within $m$ positions of each other.
    In an m-dependent sequence, if $I, J \subset 1, 2, \ldots$ and
    ${\rm max}(I) < {\rm min}(J) - m$, then $X[I]$ and $X[J]$ must be
    independent.

- The
  [Law of Large Numbers (LLN)](https://en.wikipedia.org/wiki/Law_of_large_numbers)
  states that the averages of increasingly long subsequences converge to the
  expected value. The most basic form of the LLN applies to IID sequences. If we
  write $\bar{X}_n = (X_1 + \cdots + X_n)/n$ then the LLN states that
  $\bar{X}_n \rightarrow E[X_1]$. Exactly what is meant by the "convergence"
  $\rightarrow$ involves a discussion of
  [modes of convergence](https://en.wikipedia.org/wiki/Modes_of_convergence)
  that we will not cover further here. Also note that in an IID sequence every
  term has the same expected value, so the LLN states that the sample means
  converge to $E[X_j]$ for any $j$.

- The
  [Central Limit Theorem (CLT)](https://en.wikipedia.org/wiki/Central_limit_theorem)
  states that the distribution of $\sqrt{n}(\bar{X}_n - \mu)$ tends to a normal
  distribution with mean $0$ and variance $\sigma^2$. Here we limit our
  discussion to IID sequences, so that $\mu = E[X_i]$ and
  $\sigma^2 = {\rm Var}[X_i]$ for any $i$. One implication of this result is
  that the sample means (which are natural estimators of the population mean)
  have estimation errors on the order of $\sqrt{n}$, since we have to "blow up"
  the difference between the sample and population means by a factor of
  $\sqrt{n}$ to overcome the concentrating effect of the LLN. The CLT is a
  fundamentally important tool in probability theory and statistics, and there
  are many versions of the CLT that apply in different situations.

- There are many additional tools in probability theory that are used to obtain
  more subtle results along the lines of the LLN and CLT. We will not discuss
  this more here, except to note the very important roles of the
  [Markov inequality](https://en.wikipedia.org/wiki/Markov%27s_inequality),
  [Chebyshev's inequality](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality),
  and the
  [Bernstein inequality](<https://en.wikipedia.org/wiki/Bernstein_inequalities_(probability_theory)>).

### Stochastic processes

- A [stochastic process](https://en.wikipedia.org/wiki/Stochastic_process) is a
  random object indexed by a variable $t$ that is well-ordered. Usually $t$ is
  either integer-valued or real-valued.

- The domain for the index $t$ is usually infinite, either countable (in the
  case of integers) or uncountable (in the case of a real index). Thus
  stochastic processes lie in infinite dimensional vector spaces, which
  introduces many issues that are not present in the case of finite dimensional
  random vectors.

- A
  [finite dimensional distribution](https://en.wikipedia.org/wiki/Finite-dimensional_distribution)
  of a stochastic process $Y$ is $Y[T] = [Y[T_1], \ldots, Y[T_m]]$, where $T$ is
  a fixed sequence of $m$ index values.

- A [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) is a
  stochastic process whose finite-dimensional distributions are Gaussian.
  [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion) is a Gaussian
  process with continuous sample paths.

## Linear algebra

- A [vector space](https://en.wikipedia.org/wiki/Vector_space) over the real
  numbers is a collection of abstract objects that can be added together, and
  that can be scaled by (real) numbers. It is important to keep in mind the
  distinction between the _vectors_ in a vector space and the _scalars_, which
  for our purposes will always be the real numbers.

  - All vector spaces must contain an element denoted $0$ that has certain
    properties as stated below.

  - Addition and scalar multiplication must satisfy the following axioms, for
    vectors $x$, $y$ and real scalars $c$, $d$:

    - $x + y = y + x$; $(x + y) + z = x + (y + z)$, $0 + x = x + 0 = x$;
      $(-x) + x = x + (-x) = 0$.

    - $0x = 0$; $1x = x$; $c(dx) = (cd)x$.

    - $c(x+y) = cx + cy$; $(c+d)x = cx + dx$

- A basic example of a vector space is the set of all "k-tuples". For example,
  take $k=2$, so a 2-tuple has the form $[a, b]$, e.g. $[1, 0]$ or $[5,
  -4]$.
  These can be added component-wise so that, e.g. $[3, 4] + [1, 2] =
  [4, 6]$.
  These can be scaled so that, e.g. $3\cdot [4, 5] = [12, 15]$. One can verify
  that the axioms stated above hold for this vector space.

- The vector space of $k$-tuples with real entries is denoted ${\mathbb R}^k$.
  We will call $k$ the _dimension_ of ${\mathbb R}^k$ but are not defining this
  term formally yet.

- Another example of a vector space is the set of continuous real-valued
  functions of a real variable.

- Given a collection of vectors $v_1, \ldots, v_m$, a _linear combination_ of
  these vectors is a vector of the form $c_1v_1 + \cdots + c_mv_m = 0$, where
  the $c_i$ are scalars. Linear combinations of vectors play a very important
  role in linear algebra. Here are some key properties of linear combinations:

  - A collection of vectors $v_1, \ldots, v_m$ in a vector space has the
    property of
    [linear independence](https://en.wikipedia.org/wiki/Linear_independence) if
    for any scalars $c_1, \ldots, c_m$ such that $c_1v_1 + \cdots + c_mv_m = 0$,
    it follows that $c_1, \ldots, c_m = 0$. That is, any linear combination of a
    set of linearly independent vectors is zero if and only if all of the scalar
    coefficients are zero.

  - A maximal set of linearly independent vectors in a vector space is called a
    [basis](<https://en.wikipedia.org/wiki/Basis_(linear_algebra)>).

  - The _span_ of a collection of vectors is the set of all vectors that can be
    expressed as linear combinations of them. A basis has the property that its
    span is the entire vector space.

  - The size (number of elements) in any basis is the same, and this common
    value is called the
    [dimension](<https://en.wikipedia.org/wiki/Dimension_(vector_space)>) of the
    vector space.

  - If a vector space has dimension $d$, then given a collection of $q < d$
    linearly independent vectors $u_1, \ldots, u_q$, the span of the $u_j$ is a
    $q$-dimensional [subspace](https://en.wikipedia.org/wiki/Subspace) of the
    vector space.

- Inner products and norms

  - The _dot product_ or
    [inner product](https://en.wikipedia.org/wiki/Inner_product_space) between
    two vectors is a mapping that takes two vectors and yields a scalar. It must
    satisfy certain properties such as
    $\langle x, y\rangle = \langle y, x\rangle$,
    $\langle x+y, z \rangle = \langle x, z \rangle + \langle y, z\rangle$, and
    $\langle cx, y\rangle = c\langle x, y\rangle$, where $x$, $y$, $z$ are
    vectors and $c$ is a scalar.

  - If we are working with the vector space of k-tuples, then the canonical dot
    product is formed as $\langle x, y\rangle = \sum_j x_j y_j$. Other possible
    dot products on this space would, for example, have the form
    $\langle x, y\rangle = \sum_j w_jx_j y_j$ for non-negative scalars $w_j$.

  - A [norm](<https://en.wikipedia.org/wiki/Norm_(mathematics)>) on a vector
    space is a mapping from the vectors to the non-negative reals. It is a way
    of defining the length or magnitude of a vector. A dot product always yields
    a norm via $\\|x\\|^2 = \langle x, x\rangle$. All norms have the following
    properties:

    - Triangle inequality: $\|x + y\| \le \|x\| + \|y\|$

    - Homogeneity: $\\|c\cdot x\\| = \|c\|\cdot \\|x\\|$

    - Positiveness: $\\|x\\| = 0$ implies that $x=0$

  - A very fundamental result is the
    [Cauchy-Schwarz inequality](https://en.wikipedia.org/wiki/Cauchy-Schwarz_inequality)
    $|\langle x, y\rangle| \le \\|x\\|\cdot \\|y\\|$.

- Linear transformations

  - A [linear transformation](https://en.wikipedia.org/wiki/Linear_map) is a
    mapping from one vector space to another, or from a vector space to itself.
    A linear transformation $T$ must satisfy $T(cx) = cT(x)$ and
    $T(x+y) = T(x) + T(y)$, for a scalar $c$ and vectors $x$ and $y$.

  - For most of the rest of this document, we focus on vector spaces consisting
    of k-tuples of real numbers, i.e. ${\mathbb R}^k$.

  - A [matrix](<https://en.wikipedia.org/wiki/Matrix_(mathematics)>) is an array
    of numbers, with $r$ rows and $c$ columns.

  - A _column vector_ is a matrix with $1$ column, i.e. an $r\times 1$ matrix. A
    _row vector_ is a $1\times c$ matrix. A vector that is not specified to be
    either a row vector or a column vector can usually be taken to be a column
    vector.

  - A $r\times c$ matrix can be multiplied (on the right) with a $c$ dimensional
    vector, yielding a $r$-dimensional vector. Let $M$, $x$ denote such a matrix
    and vector, and let $y = Mx$. Then $y_i$ is the dot product of the
    $i^{\rm th}$ row of $M$ with $x$. This is called _matrix-vector
    multiplication_.

  - Via matrix-vector multiplication, a matrix represents a linear
    transformation. Specifically, a $r\times c$ matrix is a transformation from
    ${\mathbb R}^c$ to ${\mathbb R}^r$.

  - We can multiply matrices together, and this corresponds to the _composition_
    of the linear transformations represented by the matrices. Recall that
    _composing_ two functions $f$ and $g$ yields the function $h(x) = f(g(x))$.
    This only makes sense when the range of $g$ is contained within the domain
    of $f$. If the matrices $A$ and $B$ represent linear transformations $T_A$
    and $T_B$, then the matrix product $A\cdot B$ represents the composition
    $T_A(T_B(x))$.

  - To multiply two matrices together, the matrices must have corresponding
    dimensions, i.e. to take the product of $A$ and $B$, the the number of
    columns of $A$ must be equal to the number of rows of $B$. Then, the product
    $AB$ can be formed from dot products, specifically, the element $i$, $j$ of
    $AB$ is the dot product of the $i^{\rm th}$ row of $A$ with the $j^{\rm th}$
    column of $B$.

  - A linear transformation between two vector spaces of the same dimension may
    be [invertible](https://en.wikipedia.org/wiki/Invertibility).
    Transformations between spaces of differing dimension can never be
    invertible. A transformation is invertible if $T(x) = T(y)$ always implies
    that $x = y$. When this occurs it is possible to construct another linear
    transformation denoted $T^{-1}$ such that $T(T^{-1}(x)) = x$ and
    $T^{-1}(T(x)) = x$. If $M$ is a matrix representing the linear
    transformation $M$ then $M$ is invertible if there exists another matrix
    $M^{-1}$ such that $MM^{-1} = M^{-1}M = I$, where $I$ is the
    [identity matrix](https://en.wikipedia.org/wiki/Identity_matrix).

- Orthogonality and projections

  - A linear transformation is
    [orthogonal](https://en.wikipedia.org/wiki/Orthogonal_transformation) if it
    preserves the inner-product, i.e.
    $\langle x, y\rangle = \langle T(x), T(y)\rangle$ for all elements $x$, $y$
    in the vector space.

  - An orthogonal transformation is norm-preserving, in that
    $\\|x\\| = \\|T(x)\\|$ for all $x$.

  - If we represent an orthogonal transformation $T$ with a matrix $M$ (acting
    via left-multiplication), then the matrix $M$ is orthogonal in the sense
    that $M^\prime M = I$. If $M$ is square then it also follows that
    $MM^\prime = I$, but in general a matrix being orthogonal only implies that
    $M^\prime M = I$.

  - Given a subspace ${\cal S}$ of a vector space ${\cal T}$, there exists an
    _orthogonal complement_ ${\cal S}^\perp$ which consists of all vectors
    $v\in {\cal T}$ such that $\langle v, u\rangle = 0$ for all
    $u \in {\cal S}$.

  - Suppose we have a $d$-dimensional vector space ${\cal T}$ and a
    $q$-dimensional subspace ${\cal S}$ of it. For any vector $v\in {\cal T}$,
    there is a unique vector $s \in {\cal S}$ that is closest to $v$, i.e. it
    minimizes the distance $\\|s - v\\|$. This is called the
    [projection](<https://en.wikipedia.org/wiki/Projection_(linear_algebra)>) of
    $v$ onto ${\cal S}$.

    - Projection is a linear transformation.

    - Every vector $v$ can be uniquely written as the sum $u + u_\perp$, where
      $u \in {\cal S}$ and $u_\perp \in {\cal S}^\perp$.

- The Singular Value Decomposition

  - Let $X$ denote an arbitrary $n \times p$ matrix, where without loss of
    generality $p < n$. The matrix $X$ can always be factored as
    $X = USV^\prime$, where $U$ is $n \times p$, $S$ is $p \times p$, and $V$ is
    $p \times p$. Further, $U$ is orthogonal ($U^\prime U = I$), $V$ is square
    and orthogonal ($V^\prime V = VV^\prime = I$), and $S$ is diagonal with
    non-increasing positive diagonal values, i.e.
    $S
    = {\rm diag}(S_{11}, \ldots, S_{nn})$ with
    $S_{11} \ge S_{22} \ge
    \cdots$.

  - If we view $X$ as a linear transformation from
    ${\mathbb R}^p \rightarrow {\mathbb R}^n$, then the SVD shows that the
    action of $X$ involves (i) an orthogonal change of coordinates on
    ${\mathbb R}^p$, expressed by $V$, (ii) scaling along each axis, expressed
    by $S$, and (iii) an orthogonal change of coordinates on ${\mathbb R}^n$,
    expressed by $U$.

  - The SVD has many important applications in data analysis, mostly due to its
    ability to produce _low rank approximations_ to $X$. If we truncate the SVD
    to the first $j$ terms, i.e. let $U^{(j)}$ represent the $n\times j$ matrix
    consisting of the first $j$ columns of $U$, $S^{(j)}$ represent the
    $j\times j$ upper left submatrix of $S$, and $V^{(j)}$ represent the
    $d\times j$ matrix consisting of the first $j$ columns of $V$, then
    $X^{(j)} \equiv U^{(j)}S^{(j)}V^{(j)\prime}$ is the best possible rank $j$
    approximation to $X$. This fact is the basis for the
    [Eckart-Young theorem](https://en.wikipedia.org/wiki/Low-rank_approximation).

- Eigendecompositions and invariant spaces

  - An [invariant subspace](https://en.wikipedia.org/wiki/Invariant_subspace)
    for a linear transformation $T$ is a subspace ${\cal S}$ such that
    $T(x) \in {\cal S}$ for all $x\in{\cal S}$. The subspace spanned by $0$ and
    the entire vector space are always invariant subspaces.

  - A one-dimensional invariant subspace is spanned by a vector $v$ such that
    $T(v) = \lambda v$. That is, $T$ acts by scaling on $v$. Such a vector $v$
    is called an
    [eigenvector](https://en.wikipedia.org/wiki/Eigenvectors_and_eigenvalues)
    and the value of $\lambda$ is its associated _eigenvalue_.

  - It is very useful to be able to identify all invariant subspaces of a given
    linear transformation $T$. Unfortunately the general situation is quite
    complex, in that $T$ may have no (nontrivial) invariance subspaces.

  - When $T$ is symmetric the situation is much more favorable.

    - A symmetric transformation (or matrix) has only real eigenvalues.

    - It is always possible to construct a basis of eigenvectors for a symmetric
      matrix. That is, for a linear transformation on a $d$-dimensional vector
      space, we can construct a sequence
      $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d$ and associated
      eigenvectors $v_1, \ldots, v_d$ such that (i) $Tv_j = \lambda v_j$ for all
      $j$, (ii) $v_1, \ldots, v_d$ are a basis for ${\mathbb R}^d$, and (iii)
      the $v_j$ are mutually orthogonal.

    - The
      [Schur decomposition](https://en.wikipedia.org/wiki/Schur_decomposition)
      states that a symmetric matrix $A$ can be written in the form
      $Q\Lambda Q^\prime$, where $Q$ is orthogonal and $\Lambda$ is diagonal.
      The elements of $\Lambda$ are the eigenvalues of $A$ and the columns of
      $Q$ are the corresponding eigenvectors.

- The determinant and trace

  - The [determinant](https://en.wikipedia.org/wiki/Determinant) is a real
    number characteristic of a square matrix. One of several equivalent
    definitions of the determinant is that it is the product of the eigenvalues.

  - A square matrix is invertible if and only if all of its eigenvalues are
    non-zero, which in turn holds if and only if its determinant is non-zero.

  - The determinant of a linear transformation has important geometric
    properties, in that it defines how volumes change under mapping by the
    transformation. If $A$ is a matrix representing a linear transformation and
    $S$ is the unit hypercube, then the volume of
    $\\{A(s) \; | \; s\in {\cal S}\\}$ is ${\rm det}(A)$ times the volume of
    ${\cal S}$.

  - The determinant of an orthogonal matrix is $1$.

  - The determinant is multiplicative with respect to matrix multiplication:
    ${\rm det}(AB) = {\rm det}(A)\cdot{\rm det}(B)$.

  - The [trace](https://en.wikipedia.org/wiki/Trace) is a real number
    characteristic of a square matrix defined to be the sum of the diagonal
    elements.

  - The trace exhibits cyclic invariance:
    ${\rm tr}(ABC) = {\rm tr}(CAB) = {\rm tr}(BCA)$, as long as the dimensions
    are such that the matrix products are defined.

  - The trace is the sum of the eigenvalues.

- Quadratic forms

  - If $T$ is a linear transformation from ${\mathbb R}^d$ to ${\mathbb R}^d$,
    then the mapping $x \rightarrow \langle T(x), x\rangle$ is called a
    [quadratic form](https://en.wikipedia.org/wiki/Optimization). In matrix
    notation, this takes the form $x \rightarrow x^\prime A x$, where $A$ is a
    square matrix.

  - Without loss of generality $A$ is symmetric, since $A$ and
    $(A + A^\prime)/2$ yield the same quadratic form.

  - $A$ is a
    [positive semidefinite matrix](https://en.wikipedia.org/wiki/Definite_matrix)
    if $x^\prime A x \ge 0$ for all $x$. $A$ is a _positive definite matrix_ if
    $x^\prime A x > 0$ for all $x\ne 0$.

  - Since $A$ is symmetric, it has a full set of real eigenvalues, and $A$ is
    positive-semidefinite if and only if all of these eigenvalues are
    non-negative. Further, $A$ is positive definite if and only if all its
    eigenvalues are strictly positive.

  - A covariance matrix is always positive semidefinite. Further, if $X$ is a
    random vector of length $d$ and $v$ is a fixed vector of length $d$, and $C$
    is the $d\times d$ covariance matrix of $X$, then the variance of
    $\langle v, X\rangle$ is equal to $v^\prime C v$.

## Optimization

- Many data analysis tasks involve
  [optimization](https://en.wikipedia.org/wiki/Optimization). A very common
  example is fitting models to data, but there are other important roles that
  optimization plays in data analysis as well.

- Formally, a mathematical optimization problem involves an _objective function_
  $f$ defined on a domain $\Omega$ that takes on values in ${\mathbb R}$. The
  goal of _unconstrained optimization_ is to find $x\in \Omega$ that minimizes
  $f$.

- If we wish to maximize $f$ we can minimize $-f$.

- If $f$ has two continuous derivatives, we can make use of calculus. For
  unconstrained minimization, the minimizer must occur at a
  [stationary point](https://en.wikipedia.org/wiki/Stationary_point),
  $\partial f/\partial x = 0$. However it is common that the
  [gradient](https://en.wikipedia.org/wiki/Gradient) is nonlinear and the
  stationary point cannot be obtained in analytic form. This converts the
  problem of numerical minimization to that of numerically solving a system of
  nonlinear equations.

- Not all optimization problems have a unique solution. Some problems have
  multiple solutions, i.e. there are multiple distinct $x$ that minimize $f$. In
  other cases there may be a sequence of points $x_i$ such that $f(x_i)$
  approaches a minimizer $f$ but does not achieve the minimum on $\Omega$ (or on
  ${\cal S}$).

### Constrained optimization

- In _constrained optimization_ there is a set ${\cal S} \subset \Omega$ and the
  goal is to minimize $f$ subject to the constraint $x \in {\cal S}$. In
  principal we can just define $f(x) = \infty$ when $x \notin {\cal S}$ and
  treat the problem as being unconstrained. But this breaks algorithms that
  depend on the smoothness or convexity of $f$. Therefore it often makes sense
  to approach constrained optimization directly rather than by recasting the
  problem as an unconstrained one.

- For constrained minimization we consider how the constraints are defined. In
  many cases the constraint set is defined by a combination of _equality
  constraints_ ($g(x) = 0$) and _inequality constraints_ ($g(x) > 0$).

- If there are only equality constraints, then the method of
  [Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier) can
  be used.

- In the more general case with both equality and inequality constraints, the
  [KKT conditions](https://en.wikipedia.org/wiki/Karush-Kuhn-Tucker_conditions)
  define a solution.

### The one-dimensional case

- The case where $f$ is defined on an interval $[a, b]$ of the real line is
  special. We can trap a minimum in a _bracket_, which is a triple of values
  $x_1 < x_2 < x_3$ such that $f(x_2) < {\rm min}(f(x_1), f(x_3))$. We can then
  successively squeeze the bracket by testing the midpoints $(x_1 + x_2)/2$ and
  $(x_2 + x_3)/2$. One of these midpoints can always be used to create a new
  bracket of lesser width than the preceding bracket. This is called the
  [bisection method](https://en.wikipedia.org/wiki/Bisection_method).

### Newton's method

- A multivariate quadratic function has the form
  $f(x) = c + b^\prime x + x^\prime A x$. If $A$ is invertible we can complete
  the square to obtain the equivalent form
  $f(x) = (x + A^{-1}b/2)^\prime A (x + A^{-1}b/2) - c$. When $A$ is a positive
  semidefinite (PSD) matrix, this expression is minimized when
  $x + A^{-1}b/2 = 0$ so $x = -A^{-1}b/2$.

- A smooth function $f(x)$ can be
  \[approximated\]((https://en.wikipedia.org/wiki/Taylor%27s_theorem) local to a
  given point $x_0$ with the approximation

  $f(x) \approx f(x_0) + (x - x_0)^\prime \nabla_f(x_0) + (x-x_0)^\prime
  H_f(x_0)(x-x_0)/2$

  where $\nabla_f$ is the
  [gradient function](https://en.wikipedia.org/wiki/Gradient) and $H_f$ is the
  [Hessian](https://en.wikipedia.org/wiki/Gradient). If we treat this quadratic
  approximation as if it were exact, we would find the minimizer to be
  $x = x_0 - H_f(x_0)^{-1}\nabla_f(x_0)$.

- This gives rise to an iterative algorithm for numerically minimizing $f$, by
  iteratively setting $x\_{i+1} = x_i - H_f(x_i)^{-1}\nabla_f(x_i)$. This is
  called
  [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization).

- Whether Newton's method converges may depend on the starting values. If it
  does converge it converges very quickly.

- Newton's method relies on the second derivative (Hessian) matrix, so it is a
  _second order_ method.

- Due to the uncertain convergence and difficulty in computing the Hessian
  matrix, Newton's method is used much less than some other _first order_
  methods that do not require calculation of the Hessian matrix.

### Convexity

- A function is strictly [convex](https://en.wikipedia.org/wiki/Convex_function)
  if every secant line lies strictly below the function. Formally, this means
  that for any points $x \ne y$ in the domain of the function,

  $f(\lambda x + (1-\lambda)y) \< \lambda f(x) + (1 - \lambda)f(y)$

  for all $0 \< \lambda \< 1$.

- A strictly convex function can have at most one local minimum, and if a local
  minimum exists it is also a global minimum.

- If a function has two continuous derivatives, it is strictly convex if and
  only if its Hessian matrix is positive definite for all $x$.

- A function is strongly convex if the dterminant of its Hessian matrix is
  bounded away from zero, that is, there exists $\epsilon>0$ such that
  $\\|H_f(x)\\| \ge \epsilon > 0$ for all $x$.

- A strongly convex function on ${\mathbb R}^d$ is guaranteed to have a unique
  global minimum.

### Gradient descent

- The gradient $\nabla_f$ of the objective function points in the direction in
  the domain of the function in which the function increases fastest. Therefore,
  letting $\lambda \in {\mathbb R}$, the restricted function
  $f(x + \lambda\nabla_f(x))$ is increasing in $\lambda$ for sufficiently small
  $\lambda>0$.

- Since we want to minimize the objective function, we can consider
  $f(x - \lambda \nabla_f(x))$. For sufficiently small $\lambda>0$,
  $f(x - \lambda\nabla_f(x)) < f(x)$. Either a
  [line search](https://en.wikipedia.org/wiki/Line_search) or a fixed step size
  may be used to achieve a sequence of iterates that should (in some cases)
  minimize the objective function.

- Basic gradient descent often converges quite slowly, especially if the Hessian
  matrix has a large condsition number. A major breakthrough was the discovery
  of
  [conjugate gradient methods](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
  in the 1950's. We discuss these in the next section.

- This area has had a revival over the last 10 years since a form of gradient
  descent is used to fit deep learning models. Specifically, there have been
  several major new ideas in the area of
  [stochastic gradient descent](https://en.wikipedia.org/wiki/Conjugate_gradient_method),
  but we will not cover these further here.

### Conjugate gradient methods

- Suppose that $H_f$ is the Hessian matrix of the objective function, and we aim
  to minimize $f$ by conducting
  [line searches](https://en.wikipedia.org/wiki/Line_search) over a sequence of
  _search directions_. It turns out to be desirable for these search directions
  to be _conjugate_, meaning that they are orthogonal with respect to $H_f$.

- For a quadratic function on a domain of dimension $d$, searching along
  conjugate directions will yield the exact solution to the optimization problem
  in $d$ line searches. That is, $d$ line searches is equivalent to one Newton
  step (since Newton's method converges exactly in one step for quadratic
  functions).

- Since smooth functions are approximately quadratic (via Taylor expansion),
  conjugate directions with respect to the Hessian matrix are generally much
  better search directions than non-conjugate directions.

- Directly constructing a conjugate basis would require explicit construction of
  the Hessian matrix $H_f$, which would yield a second-order method, in which
  case it might be just as well to use Newton's method.

- The key insight behind the conjugate gradient methods is that a set of
  conjugate directions can be constructed sequentially, using line searches but
  without requiring explicit calculation of the Hessian matrix. Hence this
  yields a first-order method.

### Quasi-Newton methods

- [Quasi-Newton](https://en.wikipedia.org/wiki/Quasi-Newton_method) methods are
  first-order methods that approximate Newton's method by using an approximation
  to the Hessian matrix. The approximate Hessian is obtained over a sequence of
  line searches, using the gradients that are calculated at each step of the
  optimization.

- Suppose we have two points $x_i$, $x_{i+1}$ and their gradients $g_i$,
  $g_{i+1}$. The _secant equation_ $g_{i+1} - g_i \approx H(x_{i+1}-x_i)$ is
  used to update the Hessian approximation $H$.

- The most popular algorithm in this class is arguably
  [LBFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS), which stands for
  "limited memory Fletcher-Goldfarb-Shanno" algorithm.

### Global/combinatorial optimization

- For smooth functions, it is almost always advantageous to make use of the
  information in the gradient. However for some highly irregular functions, or
  for non-differentiable functions or discrete domains, gradient methods are not
  applicable. Further, gradient methods tend to converge to local modes when a
  function has many optima. For these reasons a large collection of methods that
  do not employ gradient descent have been developed.

## Statistical inference

- Statistical inference refers to all aspects of learning from data. Here we
  consider statistical inference using probability models. The standard process
  is to fit a probability model to data, and then use the probability model to
  answer questions about the
  [population](https://en.wikipedia.org/wiki/Statistical_population) from which
  the data were sampled.

- The premise that the observed data are a
  [sample](<https://en.wikipedia.org/wiki/Sampling_(statistics)>) from the
  population is central to this framework.

- In the most common setting for statistical inference, we consider a family of
  probability models $P\_\theta$ indexed by a _parameter_ $\theta$. The
  parameter is usually constrained to lie in a _parameter space_ $\Theta$. This
  setting is referred to as
  [frequentist inference](https://en.wikipedia.org/wiki/Frequentist_inference).
  The most commonly encountered non-frequentist approach to inference is
  [Bayesian inference](https://en.wikipedia.org/wiki/Bayesian_inference).

- In the most basic setting, the parameter is a single scalar value.
  Finite-dimensional vector-valued parameters are also commonly encountered. In
  some cases when there are multiple parameters they can be partitioned into the
  parameters of primary interest and the
  [nuisance parameters](https://en.wikipedia.org/wiki/Nuisance_parameter).

- A parameter may be of infinite dimension, even of
  [uncountable](https://en.wikipedia.org/wiki/Uncountable_set) dimension. In the
  latter case the class of models may be referred to as _nonparametric_ although
  this term is somewhat fungible.

- In the classical setting, the class $P_\theta$ of probability models is fixed
  and it is possible to sample data from it of various sizes. In some more
  modern settings, the class of models is directly constructed to vary (grow in
  size) with increasing data set size, so can be denoted $P_{\theta_n}$.

### Parameter estimation

- Most statistical inference begins with applying a scheme to estimate the
  parameters from the data. Formally, an _estimator_ is a function of the data.

- Any function of the data can be an estimator. But there are some ways of
  generating estimators that can be applied in many common settings. We briefly
  describe two of the most common here:

  - [Method of moments](<https://en.wikipedia.org/wiki/Method_of_moments_(statistics)>):\_
    In this approach one identifies functions of the data $m(D)$ whose expected
    values can be calculated and are functions of the parameters. This gives us
    equations of the form $E[m(D)] = g(\theta)$, which in turn yields
    [estimating equations](https://en.wikipedia.org/wiki/Estimating_equations)
    $m(D) - g(\theta) = 0$. These equations can in principle be solved to yield
    estimates of the parameters.

    - If there is a single parameter, we can use a single estimating equation.

    - If there are $q$ parameters, we need at least $q$ moment equations
      $E[m_j(D)] = g_j(\theta)$, $j=1,\ldots, q$ to identify the parameters.

    - Using the
      [generalized method of moments (GMM)](https://en.wikipedia.org/wiki/Generalized_method_of_moments),
      it is possible to have more estimating equations than the number of
      parameters, but the parameters will not be identified if we have fewer
      estimating equations than the number of parameters.

    - The method of moments does not require specification of a _data generating
      model_, and therefore may have fewer assumptions than approaches based on
      a specified data generating model.

    - A very basic example of the method of moments is estimation of the
      variance $\sigma^2$. Estimating the variance requires us to also estimate
      the mean $\mu$. The parameter is thus a two-dimensional vector
      $\theta = (\mu, \sigma^2)$. One way to set up the estimation is to define
      two moment expressions: $m_1 = X_1 + \cdots + X_n$ and
      $m_2 = X_1^2 + \cdots + X_n^2$. Their corresponding expected values are
      $g_1 = n\mu$ and $g_2 = n(\mu^2 + \sigma^2)$. These equations can be
      solved to yield method of moment estimators
      $\hat{\mu} = (X_1 + \cdots + X_n)/n$ and
      $\hat{\sigma}^2 = ((X_1-\hat{\mu})^2 + \cdots + (X_n-\hat{\mu})^2)/n$.

  - [Maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood)

    - Maximum likelihood begins with the specification of a data generating
      model $P_\theta(D)$ where $\theta$ is the parameter and $D$ is the data.
      From this we obtain the
      [log likelihood](https://en.wikipedia.org/wiki/Likelihood_function)
      $L(\theta; D) = \log P_\theta(D)$.

    - The maximum likelihood estimator is ${\rm argmax}_\theta L(\theta; D)$.

    - Like any optimization problem, there may be zero, one, or multiple
      solutions to the optimization, including local and global optima, and it
      may be difficult to obtain them numerically.

    - If the log-likelihood is smooth, the MLE will be a stationary point of the
      score equations $\nabla_L(\theta) = 0$, where $\nabla_L(\theta)$ is the
      gradient of the log-likelihood $L(\theta; D)$, which in this context is
      known as the _score function_. Thus, for differentiable models finding the
      MLE is equivalent to solving the score equations, which are therefore
      estimating equations. We can compare this to the method of moments which
      also involves solving estimating equations. In some cases the estimating
      equations for maximum likelihood analysis will coincide with the
      estimating equations for the method of moments, but in general this is not
      the case.

### Sampling distributions and properties of estimators

- An estimator is a function of data, and since the data are random, an
  estimator is random. The distribution of an estimator is called the _sampling
  distribution_.

- The standard deviation of an estimator is called the _standard error_. It
  plays a very important role in frequentist inferences. It essentially
  quantifies the expected error between an estimate and its target (technically
  it is the square root of the average squared error between an estimate and its
  target).

- Several principles are utilized to judge the quality of an estimator.

  - [Bias](<https://en.wikipedia.org/wiki/Bias_(statistics)>). The bias of an
    estimator is $E[\hat{\theta}] - \theta$. An estimator is positively biased
    if its average value is greater than its target, and is negatively biased if
    its average value is less than its target. It is good for an estimator to be
    unbiased, but in practice many estimators are somewhat biased as other
    considerations besides bias (such as precision) are also important to
    consider. Bias is related to the notion of
    [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision), which
    reflects the possible presence of _systematic errors_ when estimating a
    parameter.

  - [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error): The
    mean squared error (MSE) of an estimator is $E[(\hat{\theta} - \theta)^2]$.
    It is the average squared distance between an estimator and its target. The
    root mean squared error (RMSE) is the square root of the mean squared error
    and has the advantage of having the same units (scale) as the data.

  - The variance of an estimator is defined the same way as the variance of any
    other random quantity. In the context of an estimator, variance reflects
    _precision_ -- the extent to which an estimator yields fairly similar values
    from repeated independent samples of data. Put another way, variance
    reflects random estimation errors while bias reflects systematic estimation
    errors.

  - A basic identity is that the MSE of an estimator is equal to the sum of its
    squared bias and its variance. Arguably, low MSE is the most important
    characteristic for an estimator to have, and we see from this identity that
    low MSE can be achieved by having a smaller squared bias with a larger
    variance, or a smaller variance with a larger squared bias. This is a
    reflection of a principle known as the
    [bias/variance tradeoff](https://en.wikipedia.org/wiki/Bias-variance_tradeoff).

  - [Consistency](<https://en.wikipedia.org/wiki/Consistency_(statistics)>) is a
    property of an estimator that loosely states that as the data size grows,
    the estimator converges to the true parameter value. It is an asymptotic
    property that will be discussed in more detail below.

  - [Efficiency](<https://en.wikipedia.org/wiki/Efficiency_(statistics)>) is a
    property asserting that the variance of an estimator is as low as possible.

    - We can consider the class of all unbiased estimators and ask what is the
      lowest possible variance within this class. The
      [Cramer-Rao lower bound](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound)
      provides an answer to this question.

    - We can quantify the efficiency of an estimator using the _relative
      efficiency_, which is the ratio of the variance of an estimator to the
      Cramer-Rao lower bound for this variance.

### Asymptotic analysis of estimators

- It can be difficult to exactly ascertain the properties of an estimator when
  it is applied using a fixed sample size of data. Therefore it is very common
  to characterize the limiting distribution of an estimator, and use this
  limiting distribution to understand the behavior of the estimator.

- The [delta method](https://en.wikipedia.org/wiki/Delta_method) is a key tool
  in asymptotic analysis of estimators. In its most basic form it asserts that
  when we have a sequence of random variables $X_n$ such that
  $\sqrt{n}(X_n - \theta)$ is asymptotically normal with mean zero and
  covariance $\Sigma$, then for a smooth function $g$,
  $\sqrt{n}(g(X_n) - g(\theta))$ is asymptotically normal with mean zero and
  covariance matrix $\nabla_g^\prime \Sigma \nabla_g$.

- Suppose we have a collection of $n$ independent observations. In this setting,
  many commonly-encountered estimators are
  [m-estimators](https://en.wikipedia.org/wiki/M-estimator). This is an
  estimator that arises by minimizing a sample average (or sum) of contributions
  from the observations. That is, out estimator can be written
  $\hat{\theta} = {\rm argmin}_\theta \sum_i g(\theta; x_i)$. Maximum likelihood
  is an m-estimator if the observations are independent.

- If we are using maximum likelihood analysis, the log-likelihood function for
  the overall sample is the sum of contributions from the observations:
  $L(\theta | D) = \sum L(\theta | X_i)$. The same is true of the score
  function: $s(\theta | D) = \sum s(\theta | X_i)$. When the parameter $\theta$
  is fixed at its true value, the score function has two key properties (recall
  that the score function is a random vector):

  - The expected value of the score function is zero.

  - The variance of the score function is the
    [Fisher information](https://en.wikipedia.org/wiki/Fisher_information),
    which is the negative Hessian of the log-likelihood, $-\nabla_L^2(x)$, or
    (equivalently), the negative
    [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) of
    the score function.

- Combining the results above we can conclude that the MLE is (under certain
  conditions) asymptotically normal, asymptotically unbiased, and has sampling
  covariance matrix equal to the inverse of the Fisher information matrix.
  Furthermore, this implies that the MLE achieves the Cramer-Rao lower bound and
  hence is efficient.

### Confidence sets

- A major consideration in statistical inference is to understand how much
  information we have gained (from the data) about parameters that have meaning
  in relation to a research question. When we estimate a parameter from a sample
  of data, we know that the estimate is never exactly correct. Formally, if
  $\theta$ is the parameter and $\hat{\theta}$ is the estimate of the parameter,
  we can be sure that $\hat{\theta} \ne \theta$. The _estimation error_ is
  $\hat{\theta} - \theta$, which is a random quantity.

- The standard error of a parameter gives us a sense of how far a parameter
  estimate is likely to fall from the true parameter value. Especially when
  $\hat{\theta}$ is (asymptotically) normal and unbiased, knowing the standard
  error tells us everything we can possibly know about the distribution of
  estimation errors $\hat{\theta} - \theta$.

- A [confidence interval](https://en.wikipedia.org/wiki/Confidence_interval) is
  a quantitative and explicit way to summarize the uncertainty in a parameter
  estimate.

- Formally, a confidence interval (CI) consists of two functions of the data,
  the _lower confidence bound_ (LCB) and the _upper confidence bound_ (UCB).
  Since these are functions of the data, they are random quantities and
  therefore have probability distributions. We will write $L(D)$ for the lower
  confidence bound and $U(D)$ for the upper confidence bound.

- A CI is defined in relation to a _coverage probability_, conventionally 95%,
  which is the probability that the interval covers the target (true) value of
  the parameter. If $\alpha$ is the coverage probability, then
  $\alpha = P(L(D) \le \theta \le U(D))$.

- Besides the coverage probability, the other important characteristic of a CI
  is its _average width_. Wide confidence intervals indicate that we have little
  information about the value of a parameter.

- There can be some confusion about the probabilistic interpretation of the
  probability expression $P(L(D) \le \theta \le U(D))$. The parameter $\theta$
  is a fixed quantity, but the confidence limits $L(D)$ and $U(D)$ are random.
  The randomness in this probability statement follows from the randomness in
  $L(D)$ and $U(D)$, not in $\theta$ itself.

- There are many ways to construct a confidence interval, we discuss here only
  the most basic approach that uses a
  [pivotal quantity](https://en.wikipedia.org/wiki/Pivotal_quantity). Suppose we
  have a parameter estimate $\hat{\theta}$ that is unbiased for the parameter of
  interest $\theta$, and we know (or can estimate) the standard error of
  $\hat{\theta}$, which we denote $s$. Then the expression
  $(\hat{\theta} - \theta)/s$ is a Z-score, and is a pivotal quantity. In
  particular, if $\hat{\theta}$ is (asymptotically) Gaussian, then we know that
  $P(-2 < (\hat{\theta} - \theta)/s < 2) \approx 0.95$. This expression can be
  algebraically rearranged to yield the expression
  $P(\hat{\theta} - 2s < \theta < \hat{\theta} + 2s) \approx 0.95$, which
  provides us with the (approximate) 95% CI $\hat{\theta} \pm 2s$.

- The standard error $s$ often involves nuisance parameters, and it is common to
  _plug-in_ estimates of the nuisance parameters when forming a CI. For larger
  sample sizes it is often viable to treat these estimated nuisance parameters
  as if they were actually the true values, but for small sample sizes, this
  plug-in step can have major consequences for coverage. For example, if the
  Z-score $(\hat{\theta} - \theta)/s$ follows a standard normal distribution,
  the plug-in Z-score $(\hat{\theta} - \theta)/\hat{s}$ may follow a
  distribution with heavier tails such as the
  [t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution). If
  this fact is ignored, the CI will have lower coverage probability than
  intended.

- For vector-valued parameters $\theta$, it is possible to generalize the
  concept of a confidence interval to that of a
  [confidence region](https://en.wikipedia.org/wiki/Confidence_region) (or
  confidence set). While useful, these are somewhat challenging to interpret.

### Hypothesis testing

- A framework for
  [formal testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_test) of
  hypotheses based on data was developed in the early 20th century, and
  subsequently became a major tool in applied research. While it has been
  extensively criticized, it remains an important tool in data analysis.

- The "Neyman-Pearson" framework for "null hypothesis significance testing"
  (NHST) was an important historical advance. An alternative approach was
  advocated by Fisher, and the modern approach is a synthesis of the two.

- The starting point for formal hypothesis testing is the specification of a
  null hypothesis. It is of interest to falsify this hypothesis using data.

- The null hypothesis is expressed as a statement about a parameter $\theta$
  that relates to the data generating model. A _point null hypothesis_ is a
  statement $\theta = \theta_0$. We also often encounter the setting where the
  null hypothesis is $\theta \in \Theta_0$, where $\Theta_0$ is a subset of the
  parameter space.

- We require a quantity known as the _test statistic_ $T(D)$ that summarizes the
  evidence in the data in relation to the null hypothesis. By convention, we
  construct $T$ so that larger values of $T$ reflect greater evidence against
  the null hypothesis.

- It remains to calibrate the evidence reflected in $T$. This is done by
  considering the sampling distribution of $T(D)$ when the null hypothesis is
  true. When the null hypothesis true, any evidence against the null reflected
  in the value of $T(D)$ is "by chance" or "spurious".

- Often, we are able to construct $T(D)$ so that when the null hypothesis is
  true, $T(D)$ follows a known distribution such as the standard normal
  distribution.

- Once we know the distribution of $T(D)$ under the null hypothesis, we can
  construct the [p-value](https://en.wikipedia.org/wiki/P_value) which is
  $P_0(T(D) > T(D_{\rm obs}))$, where $P_0$ indicates that the probability is
  calculated as if the null hypothesis is true. Here we use $D_{\rm obs}$ to
  represent the observed data, which may or may not follow the null
  distribution, and $D$ to represent a random dataset that follows the null
  distribution.

- There has been a lot of discussion and debate about the strengths and
  weaknesses of relying on standard errors, confidence intervals, or null
  hypothesis significance tests for statistical inference (the three pillars of
  frequentist inference). One point of view is that these approaches are more
  similar than they are different. For example, in many settings you can easily
  convert a confidence interval into a hypothesis test. If the CI does not
  contain a given null value of the parameter $\theta_0$, then you can reject
  the null hypothesis that $\theta=\theta_0$, with a p-value equal to 1 minus
  the coverage probability of the CI.

### Prediction

- A statistical model that relates two quantities, denoted $X$ and $Y$, can be
  used to make predictions of one quantity from the other. This is usually done
  using conditional distributions, which are typically estimated or
  approximated.

#### Unbiased assessment of prediction accuracy

- In the conventional setting, we observe _training data_
  $\\{(X_i, Y_i), i=1, \ldots, n\\}$ consisting of joint observations of $X$ and
  $Y$. The goal is to estimate the conditional distribution $P(Y|X)$ from the
  training data. We can then use the estimated conditional distribution
  $\hat{P}(Y|X)$ to predict $Y$ from a given value of $X$. This is often done
  using the conditional mean $E[Y|X]$. But it is also possible to use other
  conditional quantities for prediction.

- In practice, we usually take the perspective that
  ["all models are wrong but some are useful"](https://en.wikipedia.org/wiki/All_models_are_wrong),
  attributed to George Box. Thus, the estimated conditional distribution
  $\hat{P}$ is likely biased as well as subject to random variation.

- When the model is wrong, out predictions, say $\hat{E}[Y|X]$, are both biased
  and variable. We can reduce the bias by using more complex (flexible) models,
  but this increases the variability. This is a consequence of the bias/variance
  tradeoff.

- We often wish to objectively quantify the performance of a predictive model.
  Suppose that the prediction target is quantitative, and we wish to evaluate
  the root mean squared error of prediction (RMSE),
  $(E[(\hat{y} - y)^2])^{1/2}$. A naive approach is to train the model on a
  dataset, and then use this same dataset to produce an estimate of the RMSE as
  $(n^{-1}\sum_i (\hat{y}_i-y_i)^2)^{1/2}$.

- This "plug-in" estimate of the RMSE is optimistically biased (i.e. its
  expected value is smaller than the true RMSE).

- There are various ways to address this issue. We present two here.

  - We can split the training data into two subsets referred to as _training_
    and _testing_ data. The model is then trained only to the training data, and
    the RMSE is unbiasedly estimated using the testing data.

  - We can split the data randomly many times into training and testing sets,
    estimate the RMSE unbiasedly for each, and then pool these unbiased
    estimates (say by averaging). This procedure is called
    [cross validation](<https://en.wikipedia.org/wiki/Cross-validation_(statistics)>).

#### Overfitting and regularization

- Basic predictive models minimize a loss function, e.g.
  $\sum_i (y_i - \hat{y}_i)^2$. It is important to remember here that
  $\hat{y}_i = \hat{y}_i(x_i)$ is a function of $x_i$, and a model for
  $\hat{y}(x)$ has been specified.

- In the 1960's it became recognized that relying exclusively on a data-driven
  loss function yields suboptimal results. This is related to the
  [Stein paradox](https://en.wikipedia.org/wiki/Stein%27s_example).

- In modern predictive modeling, one usually combines a loss function with a
  [regularizer](<https://en.wikipedia.org/wiki/Regularization_(mathematics)>),
  which usually takes the form of a function $R(\theta)$ of the parameters that
  does not involve the data. If $L(D, \theta)$ is the loss function, then the
  parameter is estimated by minimizing $L(D, \theta) + \lambda R(\theta)$.

- Surprisingly, small gains in performance can be obtained using almost any
  regularizer, but in practice we want the regularizer to penalize against
  parameter values $\theta$ that are unlikely to be the true values of the
  parameter.
