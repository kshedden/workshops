Mediation analysis in Python Statsmodels
========================================

The approach to mediation analysis discussed here is covered in this paper:

http://imai.princeton.edu/research/files/BaronKenny.pdf

The Statsmodels source code for mediation analysis is here:

https://github.com/statsmodels/statsmodels/blob/master/statsmodels/stats/mediation.py

A notebook illustrating a mediation analysis:

http://nbviewer.jupyter.org/urls/umich.box.com/shared/static/jpmd9y99259u6dv0rj6p46993981m7zm.ipynb

Mediation analysis is a way of assessing structural hypotheses about
the relationships among variables.  It builds on regression analysis,
which aims to understand the relationship between "exposures" or
"treatments", e.g. X1, X2, and an outcome Y.  A regression analysis
might reveal how X2 relates to Y when X1 is fixed, and vice versa.  A
mediation analysis considers whether any association of X1 with Y is
"carried through" an association of X1 with X2 and an association of
X2 with Y.

If we use multiple regression to model the relationship between X1,
X2, and Y, and we compare a model regressing Y on X1 and X2 to a model
regressing Y on X1 alone, we often will find that the coefficient for
X1 becomes smaller in magnitude when including ("controlling for") X2.
Mediation analysis aims to explain why this "effect attenuation" is
occurring.

Rigorous mediation analysis is a branch of causal inference, and
involves careful specification of the assumptions (some of which are
untestable) under which mediation effects are estimable.  These issues
are important, but we will not address them in detail here.  See the
Imai et al. paper linked above for more discussion of causality and
assumptions.

Suppose we have an outcome Y, a mediator M, and an exposure T.  To be
concrete, imagine that Y is a subject's blood pressure, M is their
body mass index (BMI), and T is an indicator of whether the subject
participates in an exercise program. It is reasonable to imagine that
exercise may lead to weight loss, which in turn may lead to lower
blood pressure.  This is a "mediated" effect of exercise on blood
pressure.  At the same time, it is reasonable to imagine that exercise
has other consequences such as improving cardiovascular functioning,
that might also lead to lower blood pressure for reasons that have
nothing to do with weight loss.  These are the "direct" or
"unmediated" effects of exercise on blood pressure.

To define a mediated effect, we need to consider "counterfactual"
outcomes (also known as "potential" outcomes).  These are outcomes for
a given subject that were not actually observed, because the subject
was not seen in the setting under which the variable is defined.  In
the notation above, the counterfactual outcomes are notated as Y(t, m)
-- this is the blood pressure that would be seen for a particular
subject, had they been in treatment group t (e.g. t=1 is the exercise
program group, t=0 are the others), and had they achieved BMI m.

In a mediation setting, we view the value of the mediator variable as
a function of the treatment status, that is m = m(t).  Thus, for a
particular subject, it may be that m(1) = 30 and m(0) = 32.  This
means that this particular person would have had a BMI of 30 had they
been on the treatment arm, and 32 had they been on the control arm.

When the exposure variable T is binary, there are two "indirect" or
"mediated" effects, one for treated subjects and one for untreated
subjects.  Expressed in terms of potential outcomes, the these are
given by:

```
I1 = Y(t=1, m=m(1)) ﹣ Y(t=1, m=m(0))
I0 = Y(t=0, m=m(1)) ﹣ Y(t=0, m=m(0))
```

Each of these expressions captures what would happen to a subject
whose BMI changes as if they were treated, but whose treatment status
is held fixed, so that any other consequences of the treatment are
"blocked".

Note that we observe Y(1, m(1)) and Y(0, m(0)) in our data, but we
never observe Y(1, m(0)) or Y(0, m(1)).  This is the reason that
mediation analysis involves making untestable assumptions that allow
us to infer these unobserved values from the observed data.

Complementing the mediated (indirect) effects defined above, we have
the "direct effects", defined as

```
D1 = Y(t=1, m=m(1)) ﹣ Y(t=0, m=m(1))
D0 = Y(t=1, m=m(0)) ﹣ Y(t=0, m=m(0))
```

In these cases we are holding the mediator fixed while comparing the
outcomes under two levels of the exposure variable (T).  Thus we
"block" any mediated effect through M, and therefore see only the
"direct" effects of treatment.

The "total effect" is

```
Y(t=1, m=m(1)) - Y(t=0, m=m(0))
```

Here we allow the exposure to affect the mediator and to directly
impact the outcome.  There is only one total effect, and it can be
shown to be the average of D1 + I1 and D0 + I0.

Estimation
----------

A simple method for estimating mediation effects using linear
regression is as follows.  Fit linear models of the form

```
M = a + b*T
Y = c + d*M + e*T
```

The product `b*d` is often used to estimate the mediated effect (or
indirect effect) of M on Y.  This is called the "product of
coefficients" method.  We won't show the details here, but building on
this approach there are ways to estimate the total and direct effect.
to define the proportion of the total effect that is mediated, and to
test whether the mediated effect is nonzero (the "Sobel test").

There are two main limitations of the product of coefficients approach:

* it defines the mediation structure in terms of a specific model (two
  linear models) that may or may not fit well in a particular setting;
  in contrast, the approach defined above defines the mediation effects
  in terms of means, which does not require any particular model to
  hold

* the product of coefficients method does not hold for nonlinear
  models, e.g. logistic regression

The mediation approach implemented in Statsmodels, proposed by Imai et
al. (link to reference above), begins by specifying two regression
models, one for the mediator as a function of the exposure T and other
variables X (denoted Q1 below), and one for the outcome as a function
of the mediator M, exposure T, and other variables X (denoted Q2
below).  The models Q1 and Q2 can essentially be any type of model
capable of being fit by Statsmodels.

After fitting Q1 and Q2, create perturbed versions of the models by
randomly generating new parameter vectors based on Gaussian
approximations to the sampling distribution of the parameters.  Then
use the perturbed model Q1 to simulate potential outcomes for the
mediator variable for each subject under T=0 and T=1.

Next, plug the simulated potential values of M into the perturbed Q2,
and generate potential outcomes for the outcome variable Y.  Then,
plug these simulated potential outcomes into the defining expressions
for indirect, direct, and total effects given above.  Standard errors
for these quantities can be defined in a straightforward way, but we
do not provide details of that calculation here.

Moderated mediation
-------------------

An important notion in mediation analysis is that of "moderated
mediation".  This essentially means that not everyone in a population
experiences the mediation in the same way.  For example, in the blood
pressure example given above, it could be that that exercise is
mediated through BMI to a much greater degree for women than it is for
men.

A simple way to explore moderated mediation is by conducting the
mediation analysis separately for multiple subgroups.  A more unified
way to proceed is by including interactions with possible moderators
in the regressions discussed above.  We will not provide all the
details here, but the Statsmodels mediation procedures allows such
moderator variables to be specified.