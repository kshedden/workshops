# Probability

## Basic concepts of probability

* How do we formalize the notion of an outcome that occurs "at random"?

* The most basic example is a random outcome that can have only two
  states, e.g.  flipping a coin.  We can talk about the "probability
  of observing a head" and the "probability of observing a tail".
  This is called _discrete probability_.

 * We often wish to use probability in situations where the outcome is
   _quantitative_ (also known as _numeric_ or _continuous_).  If I scoop
   a volume of water
   from the ocean, what is the probability that I scoop exactly 300ml
   of water?  Arguably it is zero.  Actually, every possible outcome
   has probability zero, yet some outcome always occurs.

* With quantitative outcomes, we should only assign probabilities to
  ranges, not to specific outcomes, e.g. what is the probability that I
  scoop between 300ml and 310ml of water?

* It's not straightforward to use math to precisely talk about
  randomness.  Mathematical probability only became rigorous in the 20th century,
  largely due to _measure theory_.

* Key terms to understand:

  - [Sample space](https://en.wikipedia.org/wiki/Sample_space): all
    outcomes that can be observed

  - [Event](https://en.wikipedia.org/wiki/Event_(probability_theory)):
    a subset of the sample space to which we can assign a probability.
    Not all subsets are events.

  - [Probability measure](https://en.wikipedia.org/wiki/Probability_measure):
    a function that assigns a probability to each event.

* A probability measure has these properties:

  - The probability of every event is non-negative.

  - The probability of the event that includes the entire sample space
    is equal to 1.

  - If $E$ is an event and $E^c$ is its complement, then the
    probability of $E^c$ is 1 minus the probability of $E$.

  - If events $A$ and $B$ are disjoint (they contain no elements in
    common), then the probability of the union of $A$ and $B$ is the
    sum of the probability of $A$ and the probability of $B$.

* A [random variable](https://en.wikipedia.org/wiki/Random_variable) is
  a symbol such as $X$ that represents a value drawn from a
  probability distribution.  We can write expressions such as $P(X \le
  3)$, $P(X = 4)$, $P(1 \le X < 2)$, etc.  Keep in mind that $X$ has
  no fixed value, every time we view or refer to $X$ its value
  changes.

## Probability distributions

There are several effective ways to represent a probability distribution.

* If the sample space is finite, we can list the points in the sample
  space and their probabilities.  This type of probability
  distribution takes the form of a table, which is often called a
  [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function).

* If the sample space is countable, individual "atoms" (points in the
  sample space) have positive probability and there is a function
  mapping each atom to its probability.

* If the sample space is the real line, or an interval on the real
  line, we have the
  [cumulative distribution function (CDF)](https://en.wikipedia.org/wiki/Cumulative_distribution_function)
  $F(t) = P(X \le t)$.  The probabilities of all other events can be inferred
  from the CDF using the probabilities, e.g. the probability that $X$
  is between $1$ and $2$ or between $6$ and $7$ is
  $F(2) - F(1) + F(7) - F(6)$.

* If there is an atom with positive mass, the CDF is discontinuous at
  that point, e.g. $P(X=x)$ is the jump in the CDF at $x$.

* Many common distributions whose sample space is the real numbers
  have a [density](https://en.wikipedia.org/wiki/Probability_density_function),
  which is a function $f$ such that $\int_a^bf(x)dx
  = F(b) - F(a)$ is the probability that $X$ falls between $a$ and
  $b$.

* Some other ways to represent a probability distribution are its
  [quantile function](https://en.wikipedia.org/wiki/Quantile_function) and
  its [characteristic function](https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)).

## Examples of probability distributions

* The uniform distribution is either the
  [distribution](https://en.wikipedia.org/wiki/Discrete_uniform_distribution)
  in which every
  atom has the same probability (for finite sample spaces), or (for
  distributions on a real interval (a, b)), a
  [distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution) in which
  the probability of any interval is proportional to its length.

* The [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution)
  with CDF $P(X \le t) = 1 -
  \exp(-\lambda t)$ and density $f(x) = \lambda \exp(-\lambda x)$.
  The value of $\lambda$ is a _parameter_, so there are actually
  infinitely many exponential distributions, determined by the value
  of $\lambda$.

* The [normal (Gaussian) distribution](https://en.wikipedia.org/wiki/Normal_distribution)
  with density $f(x) = \exp(-(x-\mu)^2/2\sigma^2)/\sqrt{2\pi\sigma^2}$.  The values
  of $\mu$ and $\sigma$ are parameters, referring to the expected
  value and standard deviation, respectively.

## Expectation

* The [expectation](https://en.wikipedia.org/wiki/Expected_value)
  is a summary statistic of a quantitative random
  variable $X$.

* For a random variable with a density $f(x)$, the expectation $E[X]$ is
  $\int xf(x)dx$.  For a random variable with a mass function the
  expectation is $E[X] = \sum xf(x)$, where $x$ ranges over the domain of $X$.

* The expectation solves the optimization problem ${\rm argmin}_\theta
  E[(X-\theta)^2]$

* The expectation can be used to define the _center_ of a distribution,
  and has a physical anaalogy in being the balancing point if the mass
  is arranged along a beam.

* The expectation may not exist, and it may be infinite.

* The terms _mean_ and _average_ may be used synonymously with
  "expectation."

## Measures of location and dispersion

* Probability distributions are somewhat unwieldy to work with, it is
  helpful to summarize them in a few informative numbers.

* _Measures of location_ aim to capture the central value of a
  distribution.  The mean and median are measures of location, and
  there are many more.

* _Measures of dispersion_ aim to capture how spread out or
  heterogeneous are the values of the distribution.  The standard
  deviation and interquartile range are measures of dispersion, and
  there are many more.

## Quantiles and moments

Most summary statistics have one of two mathematical forms, as either
a [moment](https://en.wikipedia.org/wiki/Moment_(mathematics)) or as a
[quantile](https://en.wikipedia.org/wiki/Quantile).

* Moments are expectations of (possibly) transformed values.

  - The expected value $E[X]$ is a moment, but so is $E[X^4]$ and
    $E[\exp(X)]$.

  - The $k^{\rm th}$ raw moment is $E[X^k]$ and the $k^{\rm th}$
    central moment is $E[(X - EX)^k]$.

  - The [variance](https://en.wikipedia.org/wiki/Variance) is the second central moment $E(X - EX)^2$.  It is a
    measure of dispersion but its units are the square of the units of
    $X$.  Therefore we often use the _standard deviation_ which is the
    square root of the variance.  It has the same units as the data.

  - The $k^{\rm th}$ standardized moment is the $k^{\rm th}$ raw moment
    of the standardized variate $(X - EX)/{\rm SD}(X)$

  - The third standardized moment $E[Z^3]$ where
    $Z = (X - EX)/{\rm SD}(X)$ is a measure of _skewness_.  The fourth
    moment measures _kurtosis_, higher order moments are subtle to
    interpret.

* Quantiles are based on the inverse CDF.  If $F(t) = P(X \le t)$ is
  the CDF then the quantile function is $Q(p) = F^{-1}(p)$.  If $F$ is
  not invertible then $Q(p)$ is defined as ${\rm inf}\\{t \;|\; F(t) \ge
  p\\}$.  The $p^{\rm th}$ quantile answers the question "for a given
  $p$, what is the value $t$ such that $p$ of the mass of the
  distribution falls on or below $t$".

  - The median is a measure of location that is a quantile, it is
    $Q(1/2)$.

  - Quantiles can be used to define measures of dispersion, e.g. the
    [interquartile range](https://en.wikipedia.org/wiki/Interquartile_range)
    which is $Q(3/4) - Q(1/4)$ or the
    [median absolute deviation (MAD)](https://en.wikipedia.org/wiki/Median_absolute_deviation)
    which is the median of $|X - EX|$.

  - Quantile anaolgues of higher order moments exist, e.g. a
    quantile-based measure of skewness is

    $(Q(3/4) - 2Q(1/2) + Q(1/4)) / (Q(3/4) - Q(1/4)).$

    The theory of _L-moments_ provides a
    general means for constructing higher order summary statistics from
    quantiles.

## Conditional distributions, expectations and variances

* In most research we are working with more than one quantity at a
  time.  This leads to the concept of a
  [random vector](https://en.wikipedia.org/wiki/Multivariate_random_variable) that has a
  [joint distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution).
  Suppose that we have two jointly-distributed random variables $X$
  and $Y$, yielding a random vector $[X, Y]$.

  - The joint CDF of $X$ and $Y$ is the function
    $F(s, t) = P(X \le s, Y \le t)$.  There is also a concept of
    a joint density (which does not always exist). But
    generalizing quantiles to joint distributions is not
    straightforward.

  - If $f(x, y)$ is the joint density of $(X, Y)$, then the marginal
    density of $X$ is $f(x) = \int f(x, y)dy$, and an analogous
    result gives the marginal density of $Y$.

  - The conditional distribution $P(Y | X)$ is the distribution of
    $Y$ that results when the value of $X$ is known.  Suppose that
    there is a joint density $f(x, y)$.  Then the densithy of the
    conditional distribution $P(Y | X)$ is $f(x, y) / f(x)$.

* Conditional expectation

  - Suppose that $Y$ is quantitative and is jointly distributed with
    $X$ (which may or may not be quantitative).  The _conditional
    expectation_ of $Y$ given $X$, $E[Y|X=x]$ is (roughly speaking) the
    expected value of $Y$ when we know that $X$ takes on the value $x$.
    The rigorous definition of conditional expectation is somewhat
    different but this is the way that most people think about
    conditional expectation.

  - A special case is when $X$ has a finite sample space, so that it
    effectively partitions into groups. $E[Y | X=x]$ is the expected
    value of $Y$ when we are in group $x$.

  - The _double expectation theorem_ or _smoothing theorem_ or
    [law of total expectation](https://en.wikipedia.org/wiki/Law_of_total_expectation)
    states that
    $E[E[Y|X]] = E[Y]$.  That is, if we compute the conditional
    expectation of $Y$ at each fixed value of $X$, and then average
    these values over the (marginal) distribution of $X$, we get the
    same result as if we compute the (marginal) mean of $Y$ directly.

* Conditional variance

  - The conditional variance is analogous to the conditional
    expectation.  ${\rm Var}[Y|X=x]$ is the variance restricted to
    $(X, Y)$ values such that $X=x$.

  - The [law of total variation](https://en.wikipedia.org/wiki/Law_of_total_variance) states that

    ${\rm Var}(Y) = {\rm Var}E[Y|X] + E{\rm Var}[Y|X]$

    The term ${\rm Var}E[Y|X]$ is the _between variation_ while
    the term $E{\rm Var}[Y|X]$ is the _within variation_.  The law of
    total variation states that the overall variation is the sum of the
    between and within variations.

  - The identity can also be written as:

    $1 = {\rm Var}E[Y|X]/{\rm Var}(Y) + E{\rm Var}[Y|X]/{\rm Var}(Y)$

    This shows that the proportion of the variance in $Y$
    explained by $X$, ${\rm Var}E[Y|X]/{\rm Var}(Y)$ is complementary to
    the proportion of variance in
    $Y$ that is not explained by $X$, $E{\rm Var}[Y|X]/{\rm Var}(Y)$.

## Independence and measures of association

* Two jointly distributed random variables $X$ and $Y$ are
  [independent](https://en.wikipedia.org/wiki/Independence_(probability_theory)) if

  $P(X \in E_1 & Y \in E_2) = P(X \in E_1) \cdot P(Y \in E_2)$

  for all events $E_1$ and $E_2$.  This essentially means that knowing the
  value of $X$ tells you nothing about the value of $Y$.

* If $X$ and $Y$ are independent then $E[Y|X=x] = E[Y]$ for all values
  of $x$, and ${\rm Var}[Y|X=x] = {\rm Var}[Y]$ for all values of $x$.

* The [covariance](https://en.wikipedia.org/wiki/Covariance) is a
  measure of the relationship between $X$ and
  $Y$.  It is a moment that is defined to be $E[(X-EX)(Y-EY)]$.  The
  _correlation coefficient_ is the covariance calculated for
  standardized versions of $X$ and $Y$, that is $E[(X-EX)(Y-EY)]/({\rm
  SD}(X)\cdot {\rm SD}(Y))$.

* The correlation coefficient always lies between $-1$ and $1$.  When
  it is equal to $1$, $Y$ is a linear function of $X$ with positive
  slope.  When it is equal to $-1$ $Y$ is a linear function of $X$
  with negative slope.  If the correlation coefficient is equal to
  zero then $X$ and $Y$ are said to be _uncorrelated_.  The
  correlation coefficient is undefined if either ${\rm SD}(X) = 0$ or
  ${\rm SD}(Y) = 0$.

* The [correlation coefficient](https://en.wikipedia.org/wiki/Correlation_coefficient)
  (often called the _Pearson_ or _product
  moment_ correlation coefficient) is a _measure of association_.
  Note that the product $(X-EX)\cdot(Y-EY)$ is positive when $X$ and
  $Y$ lie on the same side of their respective expected values, and is
  greater when they both lie far on the same side of their expected
  values.  Thus, the correlation coefficient tends to be positive and
  larger when this happens frequently.

* The (Pearson) correlation coefficient is often said to be a measure
  of the _linear_ association between $X$ and $Y$, and strictly
  speaking this is true.  [Anscombe's
  quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet) shows
  many different joint distributions that have the same linear
  correlation and hence the same correlation coefficient.  However the
  correlation coefficient is able to detect many forms of dependence
  beyond that which is strictly linear.

## Limits and concentration

## Stochastic processes

* A stochastic process is a random object indexed by a variable $t$ that is well-ordered.
  Usually $t$ is either integer-valued or real-valued.

* The domain for the index $t$ is usually infinite, either countable (in the case
  of integers) or uncountable (in the case of a real index).  Thus stochastic processes
  lie in infinite dimensional vector spaces, which introduces many issues that are not
  present in the case of finite dimensional random vectors.

* A _finite dimensional_ distribution of a stochastic process $Y$ is $Y[T] = [Y[T_1}, \ldots, Y[T_m]]]$,
  where $T$ is a fixed sequence of $m$ index values.

* A _Gaussian process_ is a stochastic process whose finite-dimensional distributions are
  Gaussian.  _Brownian motion_ is a Gaussian process with continuous sample paths.

# Linear algebra

* A _vector space_ over the real numbers is a collection of abstract objects that can be added
  together, and that can be scaled by (real) numbers.  These properties
  of addition and (scalar) multiplication must satistfy the following axioms,
  for vectors $x$, $y$ and real scalars $c$, $d$:

  - $x + y = y + x$; $(x + y) + z = x + (y + z)$, $0 + x = x + 0 = x$; $(-x) + x = x + (-x) = 0$.

  - $0x = 0$; $1x = x$; $c(dx) = (cd)x$.

  - $c(x+y) = cx + cy$; $(c+d)x = cx + dx$

* A basic example of a vector space is the set of all "k-tuples".  For example, take $k=2$, so
  a 2-tuple has the form $[a, b]$, e.g. $[1, 0]$ or $[5, -4]$.  These can be added component-wise
  so that, e.g. $[3, 4] + [1, 2] = [4, 6]$.  These can be scaled so that, e.g. $3\cdot [4, 5] = [12, 15]$.
  One can verify that the axioms stated above hold for this vector space.

* The vector space of $k$-tuples with real entries is denoted ${\mathbb R}^k$.  We will call $k$ the
  _dimension_ of ${\mathbb R}^k$ but are not defining this term formally yet.

* Another example of a vector space is the set of continuous real-valued functions of a real variable.

* Inner products and norms

  - The _dot product_ or _inner product_ between two vectors is a mapping that takes two vectors and
    yields a scalar.  It must satisfy certain properties such as $\langle x, y\rangle = \langle y, x\rangle$,
    $\langle x+y, z \rangle = \langle x, z \rangle + \langle y, z\rangle$, and
    $\langle cx, y\rangle = c\langle x, y\rangle$, where $x$, $y$, $z$ are vectors and $c$ is a scalar.

  - If we are working with the vector space of k-tuples, then the dot product is formed as
    $\langle x, y\rangle = \sum_{j=1}^k x_j y_j$.

  - A _norm_ on a vector space is a mapping from the vectors to the non-negative reals.  It is a way
    of defining the length or magnitude of a vector.  A dot product always yields a norm via
    $\|x\|^2 = \langle x, x\rangle$.

  - A very fundamental result is the _Cauchy-Schwarz_ inequality $|\langle x, y\rangle| \le \|x\|\cdot \|y\|$.

* Linear transformations

  - A _linear transformation_ is a mapping from one vector space to another, or from a vector space to itself.
    A transformation $T$ must satisfy $T(cx) = cT(x)$ and $T(x+y) = T(x) + T(y)$, for a scalar $c$ and vectors
    $x$ and $y$.

  - For most of the rest of this document, we focus on vector spaces consisting of k-tuples of real
    numbers, i.e. ${\mathbb R}^k$.

  - A _matrix_ is an array of numbers, with $r$ rows and $c$ columns.

  - A _column vector_ is a matrix with $1$ column, i.e. an $r\times 1$
    matrix.  A _row vector_ is a $1\times c$ matrix.  A vector that is
    not specified to be either a row vector or a column vector can usually
    be taken to be a column vector.

  - A $r\times c$ matrix can be multiplied (on the right) with a $c$ dimensional vector, yielding a
    $r$-dimensional vector.  Let $M$, $x$ denote such a matrix and vector, and let $y = Mx$.  Then
    $y_i$ is the dot product of the $i^{\rm th}$ row of $M$ with $x$.  This is called
    _matrix vector multiplication_.

  - Via matrix vector multiplication, a matrix represents a linear transformation.  Specifically, a
    $r\times c$ matrix is a transformation from ${\mathbb R}^c$ to ${\mathbb R}^r$.

  - We can multiply matrices together, and this corresponds to the _composition_ of the linear transformations
    represented by the matrices.  Recall that _composing_ two functions $f$ and $g$ yields the
    function $h(x) = f(g(x))$.  This only makes sense when the range of $g$ is contained within the
    domain of $f$.  If the matrices $A$ and $B$ represent linear transformations $T_A$ and $T_B$, then
    the matrix product $AB$ represent the composition $T_A(T_B(x))$.

  - Two multiply two matrices together, the matrices must have corresponding dimensions, i.e. to take
    the product of $A$ and $B$, the the number of columns of $A$ must be equal to the number of rows of $B$.
    Then, the product $AB$ can be formed from dot products, specifically, the element $i$, $j$ of $AB$ is
    the dot product of the $i^{\rm th}$ row of $A$ with the $j^{\rm th}$ column of $B$.

# Optimization

## The one-dimensional case

## Convexity

## Gradient descent

## Conjugate gradients

## Global/combinatorial optimization

# Statistical inference

## Estimation

## Sampling distributions

## Hypothesis testing

## Prediction
