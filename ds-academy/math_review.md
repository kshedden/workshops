# Probability

## Basic concepts of probability

* How do we formalize the notion of an outcome that occurs "at random"?

* The most basic example is a random outcome that can have only two
  states, e.g.  flipping a coin.  We can talk about the "probability
  of observing a head" and the "probability of observing a tail".
  This is called _discrete probability_.

 * We often wish to use probability in situations where the outcome is
   quantitative (numeric, continuous).  If I scoop a volume of water
   from the ocean, what is the probability that I scoop exactly 300ml
   of water?  Arguably it is zero.  Actually, every possible outcome
   has probability zero, yet some outcome always occurs.

* With quantitative outcomes, we should only assign probabilities to
  ranges, not specific outcomes, e.g. what is the probability that I
  scoop between 300ml and 310ml of water?

* It's not straightforward to use math to precisely talk about
  randomness

* Mathematical probability only became rigorous in the 20th century,
  largely due to _measure theory_.

* Key terms to understand:

  - Sample space: all outcomes that can be observed

  - Event: a subset of the sample space to which we can assign a
    probability.  Not all subsets are events.

  - Probability measure: a function that assigns a probability to each
    event.

* A probability measure has these properties:

  - The probability of every event is non-negative.

  - The probability of the event that includes the entire sample space
    is equal to 1.

  - If $E$ is an event and $E^c$ is its complement, then the
    probability of $E^c$ is 1 minus the probability of $E$.

  - If events $A$ and $B$ are disjoint (they contain no elements in
    common), then the probability of the union of $A$ and $B$ is the
    sum of the probability of $A$ and the probability of $B$.

* The term _random variable_ is used to refer to a symbol such as $X$
  that represents a value drawn from a pronbability distribution.  We
  can write expressions such as $P(X \le 3)$, $P(X = 4)$, $P(1 \le X <
  2)$, etc.  A tricky issue is that $X$ has no fixed value, every time
  we view or refer to $X$, it's value changes.

## Probability distributions

There are several effective ways to represent a probability distribution.

* If the sample space is finite, we can list the points in the sample
  space and their probabilities.  The probability distribution is a
  table, which is often called a _probability mass function_ (pmf).

* If the sample space is countable, individual "atoms" (points in the
  sample space) have positive probability and there is a function
  mapping each atom to its probability.

* If the sample space is the real line, or an interval on the real
  line, we have the _cumulative distribution function (CDF)_ $F(t) =
  P(X \le t)$.  The probabilities of all other events can be inferred
  from the CDF using the probabilities, e.g. the probability that $X$
  is between $1$ and $2$ or between $6$ and $7$ is $F(2) - F(1) + F(7)
  - F(6)$.

* If there is an atom with positive mass, the CDF is discontinuous at
  that point, e.g. $P(X=x)$ is the jump in the CDF at $x$.

* Many common distributions whose sample space is the real numbers
  have a _density_, which is a function $f$ such that $\int_a^bf(x)dx
  = F(b) - F(a)$ is the probability that $X$ falls between $a$ and
  $b$.

* Some other ways to represent a probability distribution are its
  _quantile function_ and its _characteristic function_.

## Examples of probability distributions

* The uniform distribution is either the distribution in which every
  atom has the same probability (for finite sample spaces), or (for
  distributions on a real interval $(a, b)$), a distribution in which
  the probability of any interval is proportional to its length.

* The exponential distribution with CDF $P(X \le t) = 1 -
  \exp(-\lambda t)$ and density $f(x) = \lambda \exp(-\lambda x)$.

* The normal (Gaussian) distribution with density $f(x) =
  \exp(-(x-\mu)^2/2\sigma^2)/\sqrt{2\pi\sigma^2}$.

## Expectation

The _expectation_ is a summary statistic of a quantitative random
variable $X$.

For a random variable with a density $f(x)$, the expectation $E[X]$ is
$\int xf(x)dx$.  For a random variable with a mass function the
expectation is $\sum xf(x)$, where $x$ ranges over the domain of $X$.

The expectation solves the optimization problem ${\rm argmin}_\theta
E[(X-\theta)^2]$

The expectation can be used to define the _center_ of a distribution,
and has a physical anaalogy in being the balancing point if the mass
is arranged along a beam.

The expectation may not exist, and it may be infinite.

The terms _mean_ and _average_ may be used synonymously with
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
a _moment_ or as a _quantile_.

* Moments are expectations of (possibly) transformed values.

- The expected value $E[X]$ is a moment, but so is $E[X^4]$ and
  $E[\exp(X)]$.

- The $k^{\rm th}$ raw moment is $E[X^k]$ and the $k^{\rm th}$
  central moment is $E[(X - EX)^k]$.

- The _variance_ is the second central moment $E(X - EX)^2$.  It is a
  measure of dispersion but its units are the square of the units of
  $X$.  Therefore we often use the _standard deviation_ which is the
  square root of the variance.  It has the same units as the data.

- The $k^{\rm th}$ standardized moment is the $k^{\rm th}$ raw moment
  of the standardized variate $(X - EX)/{\rm SD}(X)$

- The third standardized moment $E[Z^3]$ where
  $Z = (X - EX)/{\rm SD}(X)$ is a measure of _skewness_.  The fourth
  moment measures _kurtosis_, higher order moments are quite subtle to
  interpret.

* Quantiles are based on the inverse CDF.  If $F(t) = P(X \le t)$ is
  the CDF then the quantile function is $Q(p) = F^{-1}(p)$.  If $F$ is
  not invertible then $Q(p)$ is defined as ${\rm inf}\{t | F(t) \ge
  p\}$.  The $p^{\rm th}$ quantile answers the question "for a given
  $p$, what is the value $t$ such that $p$ of the mass of the
  distribution falls on or below $t$".

- The median is a measure of location that is a quantile, it is
  $Q(1/2)$.

- Quantiles can be used to define measures of dispersion, e.g. the
  _interquartile range_ which is $Q(3/4) - Q(1/4)$ or the _median
  absolute deviation (MAD)_ which is the median of $|X - EX|$.

- Quantile anaolgues of higher order moments exist, e.g. a
  quantile-based measure of skewness is $Q(3/4) - 2Q(1/2) +
  Q(1/4))/(Q(3/4) - Q(1/4))$.  The theory of _L-moments_ provides a
  general means for constructing higher order summary statistics from
  quantiles.

## Conditional expectations and variances

* Conditional expectation

- In most research we are working with more than one quantity at a
  time.  This leads to the concept of a _joint distribution_.
  Suppose that we have two jointly-distributed random variables $X$
  and $Y$.  The joint CDF of $X$ and $Y$ is the function
  $F(s, t) = P(X \le s, Y \le t)$.  There is also a joint density but
  generalizing quantiles to joint distributions is not
  straightforward.

- Suppose that $X$ is quantitative and is jointly distributed with
  $Y$ (which may or may not be quantitative).  The _conditional
  expectation_ of $X$ given $Y$, $E[X|Y=y]$ is (roughly speaking) the
  expected value of $X$ when we know that $Y$ takes on the value $y$.
  The rigorous definition of conditional expectation is somewhat
  different but this is the way that most people think about
  conditional expectation.

** A special case is when $Y$ has a finite sample space, so that $Y$
   effectively partitions into groups. $E[X | Y=y]$ is the expected
   value of $X$ when we are in group $y$.

- The _double expectation theorem_ or _smoothing theorem_ states that
  $E E[X|Y] = E[X]$.  That is, if we compute the conditional
  expectation of $X$ at each fixed value of $Y$, and then average
  these values over the (marginal) distribution of $Y$, we get the
  same result as if we compute the (marginal) mean of $X$ directly.


## Categorical distributions

## Multivariate distributions

## Measures of association

## Limits and concentration

## Stochastic processes

# Linear algebra


# Optimization

## The one-dimensional case

## Convexity

## Gradient descent

## Conjugate gradients

## Global/comb inatorial optimization

# Statistical inference

## Estimation

## Sampling distributions

## Hypothesis testing

## Prediction