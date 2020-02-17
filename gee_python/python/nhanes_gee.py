# # Generalized estimating equations (GEE)

# Generalized estimating equations (GEE) is an approach to fitting
# regression models that was first formalized in the 1980's, building
# on and unifying earlier work.  The main goal of GEE is to extend the
# framework of generalized linear modeling (GLM) to handle certain
# types of dependent data.  Like GLM, GEE focuses primarily on the
# *mean structure* $E[y | x_1, \ldots, x_p]$, where $y$ here is the
# dependent variable of the regression, and $x_1, \ldots, x_p$ are the
# covariates.  As in GLM, a link function $g$ is used, so we will be
# relating the mean to the linear predictor through the mean structure
# model

# $$
# g(E[y | x_1, \ldots, x_p]) = b_0 + b_1x_1 + \cdots + b_px_p.
# $$

# A GEE regression also involves a mean/variance relationship function
# $f$, which plays the same role as in a GLM:

# $$
# {\rm Var}[y | x_1, \ldots, x_p] = \phi\cdot f(E[y | x_1, \ldots, x_p]).
# $$

# Other concepts from GLM like the scale parameter ($\phi$) and family
# (e.g. `binomial`) play similar roles in GEE as they do in GLM.

# The main aspect of a GEE that is not present in a GLM is the
# *working dependence structure*, or *working correlation*.  This
# specifies how observations are related to each other.  In a GLM, the
# observations are treated as being independent of each other, while
# GEE can accommodate many different forms of dependence between
# observations.

# The reason that the dependence structure in a GEE is referred to as
# a "working" dependence structure is that it does not have to be
# correct in order for the results of the regression to be valid.  If
# the working dependence structure is correct, the GEE results will be
# more efficient (i.e. the estimates will be more accurate, and the
# standard errors will tend to be smaller).  If the working dependence
# structure is incorrect, the parameter estimates will be less
# accurate, and the standard errors will be correspondingly larger.
# However the standard errors continue to correctly reflect the
# uncertainty in the parameter estimates, and with enough data, the
# parameter estimates will still become arbitrarily accurate.

# This notebook is organized as a case study.  We begin by using data
# from [NHANES](https://www.cdc.gov/nchs/nhanes/index.htm), the
# *National Health and Nutrition Examination Survey*.

# First we import the libraries that we will be using.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Next we read the data. For simplicity, here we will use "complete
# case analysis", meaning that we drop all cases with a missing value
# on any variable of potential interest.

# +
url = "https://raw.githubusercontent.com/kshedden/statswpy/master/NHANES/merged/nhanes_2015_2016.csv"
da = pd.read_csv(url)

# Drop unused columns, drop rows with any missing values.
vars = ["SEQN", "BPXSY1", "BPXSY2", "RIDAGEYR", "RIAGENDR", "RIDRETH1",
        "DMDEDUC2", "BMXBMI", "SMQ020", "INDFMPIR", "SDMVSTRA", "SDMVPSU",
        "DMDMARTL"]
da = da[vars].dropna()
# -

# Below we will be using several NHANES variables that are
# categorical.  To make the results easier to interpret, we next
# create variables that express the groups using text labels.

# +
da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})

da["RIDRETH1x"] = da.RIDRETH1.replace({1: "MexicanAmerican", 2: "OtherHispanic",
                   3: "NonHispanicWhite", 4: "NonHispanicBlack", 5: "Other"})

da["DMDEDUC2x"] = da.DMDEDUC2.replace({1: "Lt9", 2: "g9-11", 3: "HS", 4: "SomeCollege",
                      5: "College", 6: np.nan, 7: np.nan, 9: np.nan})

da["DMDMARTLx"] = da.DMDMARTL.replace({1: "Married", 2: "Widowed", 3: "Divorced",
                         4: "Separated", 5: "NeverMarried", 6: "Partnered",
                         77: np.nan, 99: np.nan})
# -

# ## Cluster samples and grouped data

# The overall NHANES sample for one wave is intended to represent the
# adult US population at a particular point in time (strictly
# speaking, sampling weights are required for the sample to be
# representative, but here we ignore the weights for simplicity).  To
# produce one wave of data for the NHANES study, the research staff
# visit multiple sites in the US, and sample people at each site.
# Since people who live in the same geographical region tend to be
# more alike than people living in different regions, the cluster
# sampling used in NHANES induces correlations in the data.  For this
# reason, it is not fully correct to analyze the NHANES data using a
# method like GLM that treats the cases as an independent sample.

# Below we create a "group" variable corresponding to the distinct
# combinations of a masked stratum (`SDMVSTRA`) and a masked sampling
# cluster (`SDMVPSU`), and treat this as a single level of "grouping
# structure" in the data. The reference to "masking" is due to the
# fact that for confidentiality reasons, the strata and clusters
# provided in the dataset are not the actual strata and clusters in
# the survey.  See the NHANES data documentation for more details on
# this point.

# +
da["group"] = 10 * da.SDMVSTRA + da.SDMVPSU

# Make sure we distinguish all unique values
a = da.SDMVSTRA.unique().size * da.SDMVPSU.unique().size
b = da.group.unique().size
assert(a == b)
# -

# When data are collected in groups, it is usually the case that units
# within a group are more similar than units in different groups.  Not
# accounting for this similarity has several consequences, most
# notably that the standard errors and other quantities needed for
# statistical inference are not correct.  Below we consider several
# approaches for addressing this issue.

# ## Fixed effects analysis
#
# One way to account for the relationships induced by cluster sampling
# is to include "fixed effects" for each group, i.e. to include the
# group as a covariate in the model.  Since group is categorical, it
# will automatically be converted into a series of indicator variables
# (omitting one group as the reference).  This approach can be useful,
# especially if there are relatively few groups that are large.  But
# when there are a large number of small groups, the model parameters
# can be highly inaccurate (technically, "inconsistent") when using
# fixed effects analysis.  This is called the "Neyman-Scott" problem.

# Below we will focus on regression models in which the dependent
# variable is a measure of poverty, `INDFMPIR`.  This is the ratio
# between a household's income and the poverty threshold.  Since it is
# a ratio, it makes sense to logarithmically transform it.

# Below we fit a fixed effects model using OLS:

# +
da = da.loc[da.INDFMPIR >0, :]
da["logINDFMPIR"] = np.log(da.INDFMPIR)

model = sm.OLS.from_formula("logINDFMPIR ~ RIDAGEYR + RIAGENDRx + RIDRETH1x + C(group)",
                            data=da)
result = model.fit()
print(result.summary())
# -

# There are many rows in the above output for the fixed effects
# parameters for groups.  As with any categorical covariate, one group
# was dropped and used as the reference group.  The parameter
# estimates and p-values in the table above are contrasts relative to
# the reference group.  Since the reference group was chosen
# arbitrarily, it is not very meaningful to consider the p-values.

# ## GEE

# GEE is another method for analyzing a clustered sample.  In GEE, the
# groups are viewed as inducing a correlation between observations in
# the same group.  Thus, instead of modeling the impact of the groups
# through the mean structure, as in fixed effects analysis, the group
# effect is treated as a form of correlation (dependence) structure.

# To begin, we fit a series of "marginal" mean structure models using
# OLS and GEE.  The term "marginal" here refers to the fact that the
# parameter estimates reflect average effects over the groups, as
# opposed to being conditional on a group.  For example, if we are
# interested in the difference between women and men, and we have an
# indicator variable ${\rm female}$ for female sex, with coefficient
# $b$, then $b$ represents the average difference between a female and
# a male, who could be either in the same group, or in different
# groups.

# The models below use age, gender, and ethnicity to predict family
# income.  We first fit a marginal mean structure model using linear
# least squares (OLS), ignoring the "group" information.

# +
model = sm.OLS.from_formula("logINDFMPIR ~ RIDAGEYR + RIAGENDRx + RIDRETH1x", data=da)
result = model.fit()
print(result.summary())
# -

# Next we fit the model using GEE with an "independent" working
# dependence structure.  Here, the working covariance structure states
# that observations within a cluster are independent.  As noted above,
# the working dependence structure does not need to be correct.
# However to the extent that there is correlation within groups, this
# fit will be less efficient than one that accounts for the
# dependence.

model = sm.GEE.from_formula("logINDFMPIR ~ RIDAGEYR + RIAGENDRx + RIDRETH1x", groups="group",
                            data=da)
result = model.fit()
print(result.summary())
print(result.cov_struct.summary())

# Note that the estimated regression parameters in the OLS and
# independence GEE are identical (this will always be the case), but
# the standard errors differ.  In general, the standard errors in the
# OLS fit will be too small, and GEE will give larger, and more
# correct standard errors.  Here we see that the standard errors are
# around twice as large in GEE compared to OLS.

model = sm.GEE.from_formula("logINDFMPIR ~ RIDAGEYR + RIAGENDRx + RIDRETH1x", groups="group",
                            cov_struct=sm.cov_struct.Exchangeable(), data=da)
result = model.fit()
print(result.summary())
print(result.cov_struct.summary())

# Next we use another type of working correlation structure called
# "exchangeable" that is suitable for use with grouped data.  The
# exchangeable correlation structure stipulates that any two
# observations in the same group have the same level of correlation
# between them.  This common correlation parameter can be estimated
# from the data.  Any two observations in different groups are modeled
# as being independent.  The correlation between two observations in
# the same group is called the "intraclass correlation coefficient",
# or "ICC".  We are able to see here that the estimated ICC is around
# 0.04.  The ICC ranges from 0 to 1, with an ICC of zero meaning that
# the groups are irrelevant to the outcome, and an ICC of one meaning
# that the outcome is totally determined by the groups.  While an ICC
# of 0.04 seems small, for reasons discussed below it is large enough
# to have a noticeable impact, especially when the groups are large,
# as is the case here.

# The presence of positive correlations within groups reduces the
# amount of information in the data for some purposes, and enhances
# the information for other purposes.  The impact of within-cluster
# dependence can be better understood by considering two quantities
# called the "design effect" and the "effective sample size".  These
# values quantify the loss of information when the goal is to estimate
# a mean.  When the ICC is positive, the effective sample size
# represents a hypothetical sample size (smaller than the nominal
# sample size) such that the precision for estimating a mean using an
# independent sample of size equal to the effective sample size is the
# same as would be obtained using the actual (dependent) sample.

# For example, if the ICC is zero, then the effective sample size is
# the same as the nominal sample size (the number of observed data
# values).  If the ICC is close to 1, then the effective sample size
# will become close to the number of groups (irrespective of how many
# observations are made within a group).  The design effect and
# effective sample size are calculated in the cell below.

icc = result.cov_struct.dep_params
print("ICC = %f\n" % icc)
n = da.groupby("group").size().mean()
print("Average cluster size = %f\n" % n)
de = 1 + (n - 1) * icc
print("Design effect = %f\n" % de)
ess = model.nobs / de
print("Number of observations = %f\n" % model.nobs)
print("Effective sample size = %f\n" % ess)

# In this example, the design effect is around 7, meaning that the ICC
# of around 0.04 reduces the information in our sample by around a
# factor of 7.  Since standard errors scale with the square root of
# the sample size, this means that the standard errors of the
# parameters may increase by a factor of 2-3 when properly accounting
# for the dependence structure.  This is roughly consistent with the
# results shown above, comparing the OLS standard errors to either of
# the GEE standard errors.  Note however that regression parameters do
# not behave exactly like means, so the design effect is only a rough
# indication of how the standard errors for regression parameters are
# impacted by dependence due to the clusters.

# One way to think about the intraclass correlation is to imagine that
# it is driven by unobserved cluster-level covariates.  That is, there
# are likely to be characteristics shared by all people in the same
# cluster that we do not observe.  Our failure to account for these
# covariates in the mean structure model induces intra-cluster
# dependence.  In fact, we do have access to many additional
# covariates in NHANES that could be included in the regressions, and
# we see below that as we include fewer covariates the ICC tends to
# increase, and as we include more covariates, the ICC tends to
# decrease:

model = sm.GEE.from_formula("logINDFMPIR ~ RIDAGEYR + RIAGENDRx", groups="group",
                            cov_struct=sm.cov_struct.Exchangeable(), data=da)
result = model.fit()
print(result.summary())
print(result.cov_struct.summary())

model = sm.GEE.from_formula("logINDFMPIR ~ RIDAGEYR + RIAGENDRx + RIDRETH1x + DMDMARTLx + DMDEDUC2x", groups="group",
                            cov_struct=sm.cov_struct.Exchangeable(), data=da)
result = model.fit()
print(result.summary())
print(result.cov_struct.summary())

# ## Grouped data with nonlinear mean structures

# The GEE analysis above uses a linear mean structure.  In this
# setting, GEE is equivalent to an older procedure called "Generalized
# Least Squares" (GLS).  While GEE and GLS estimate the regression
# parameters in the same way, GEE carries out inference (standard
# error calculations) using a robust approach that gives correct
# results even if the working dependence is wrong.  This robust
# inference also has a history predating GEE, and is alternatively
# known as "Huber-White" or "sandwich" covariance estimation.

# GEE can be used with any GLM link function and family, giving rise
# to nonlinear mean relationships and non-constant mean/variance
# relationships.  For example, there are GEE analogues of logistic and
# Poisson regression. In the case where the link function is not
# linear, the fitting algorithm is not least squares, but it can be
# viewed as the limit of a sequence of least squares fits, and thus
# may be referred to as "iteratively reweighted least squares".

# To demonstrate GEE with a nonlinear mean struture, we use the
# smoking status variable in NHANES, which is binary.  First, for
# reference, we fit a binomial GLM (standard logistic regression).
# This analysis does not account for the cluster (grouping) structure
# in the data, so has the potential to give misleading results.

# +
da["SMQ020x"] = da.SMQ020.replace({1: "yes", 2: "no", 7: np.nan, 9: np.nan})
dx = da[["SMQ020x", "RIDAGEYR", "RIAGENDRx", "RIDRETH1x", "group"]].dropna()
dx["Smoke"] = (da.SMQ020x == "yes").astype(np.int)

model = sm.GLM.from_formula("Smoke ~ RIDAGEYR + RIAGENDRx",
                            family=sm.families.Binomial(), data=dx)
result = model.fit()
print(result.summary())
# -

# Next we use GEE to fit the logistic regression while accounting for
# the groups.  We use here the "independence" working covariance,
# which models the data as being independent, but continues to give
# meaningful results even if the observations are not independent.
# Note that the parameter estimates are the same as obtained above
# using GLM, but the standard errors are slightly different.  In this
# case, the difference in standard errors is very minor, but in other
# cases the differences can be large.

# GEE with the independent working covariance will always give the
# same parameter estimates as GLM.  This tells us that GLM can safely
# be used with dependent data, as long as only the point estimates are
# considered.  If (as is usually the case), uncertainty is to be
# assessed, then GEE (or another appropriate approach such as mixed
# modeling) should be used.

model = sm.GEE.from_formula("Smoke ~ RIDAGEYR + RIAGENDRx",
                            groups="group",
                            family=sm.families.Binomial(), data=dx)
result = model.fit()
print(result.summary())
print(result.cov_struct.summary())

# As with the linear GEE, it is often desirable to introduce a
# non-independent covariance structure into the model.  Doing so
# provides us with some insight into the form of the dependence
# structure, and also has the potential to give more accurate point
# estimates and tighter uncertainty bounds.

# The estimates and standard errors for the fit using the exchangeable
# covariance structure are very similar to what was obtained above
# using the independence working dependence structure.  The estimated
# ICC is small, but as discsussed above, ICC's around 0.02 can have
# sizeable impact on the parameter estimates and standard errors if
# the clusters are large, as is the case here.

model = sm.GEE.from_formula("Smoke ~ RIDAGEYR + RIAGENDRx",
                            groups="group",
                            family=sm.families.Binomial(),
                            cov_struct=sm.cov_struct.Exchangeable(), data=dx)
result = model.fit()
print(result.summary())
print(result.cov_struct.summary())

# Note that while these models are estimated using the GEE framework,
# the interpretation of the effects is the same as in a GLM.  In this
# case, we are working with a logistic model, thus the parameter
# estimates shown above, say for male gender, reflect the amount by
# which the log odds for smoking is greater when comparing a male to a
# female of the same age.  More specifically, in the context of
# dependent, or grouped data, as we have here, this parameter should
# be interpreted as reflecting the average difference of log odds for
# smoking when comparing a male to a female of the same age,
# irrespective of whether they are in the same or in different groups.
# This follows from the fact that the mean structure estimated by a
# GEE is "marginal".  A related but different regression technique is
# "multilevel modeling", in which case this parameter should be
# interpreted as reflecting the contrast between a male and a female
# of the same age, who also fall into the same group.

# As discussed above in the linear mean structure example, including
# more covariates will generally reduce the ICC, but usually not to
# the point where it is negligible.

model = sm.GEE.from_formula("Smoke ~ RIDAGEYR + RIAGENDRx + RIDRETH1x",
                            groups="group",
                            family=sm.families.Binomial(),
                            cov_struct=sm.cov_struct.Exchangeable(), data=dx)
result = model.fit()
print(result.summary())
print(result.cov_struct.summary())

# In order to show that the ICC's we obtaining here are not artifacts,
# below we generate a completely random grouping variable with the
# same number of levels as the actual grouping variable, and show the
# estimated ICC values that are obtained.

for k in range(10):
    dx["group2"] = np.random.randint(0, 30, dx.shape[0])
    model = sm.GEE.from_formula("Smoke ~ RIDAGEYR + RIAGENDRx",
                                groups="group2",
                                family=sm.families.Binomial(),
                                cov_struct=sm.cov_struct.Exchangeable(), data=dx)
    result = model.fit()
    print(result.cov_struct.summary())

# ## Subject-level repeated measures

# Above we considered dependence that resulted from cluster sampling.
# Another common reason that dependence occurs is having repeated
# measures on individuals.  This happens, for example, in a
# longitudinal study, where variables of interest are measured on
# individuals repeatedly over time.  Since NHANES is a cross-sectional
# study, there are no repeated longitudinal measures.  However, there
# are repeated measures of the blood pressure variables (systolic and
# diastolic blood pressure).  Each of these variables is measured on
# every subject 2-4 times in short succession.  The reason that this
# is done is that blood pressure can be quite variable, and in
# particular can be high on an initial measurement due to subjects
# being anxious.  Thus the second and subsequent blood pressure
# measurements tend to be slightly lower than the first.

# First we need to do some data management.  The NHANES file that we
# are working with is in "wide form" (one row per subject), but for
# this analysis we need one row per blood pressure observation.  The
# Pandas `melt` method with some subsequent cleanup accomplishes this
# data restructuring.

# +
idv = ("SEQN", "BMXBMI", "SDMVSTRA", "SDMVPSU", "RIAGENDR", "RIDAGEYR", "RIDRETH1")
dx = pd.melt(da, id_vars=idv, value_vars=("BPXSY1", "BPXSY2"),
             var_name="Time", value_name="SBP")
dx["Time"] = dx.Time.apply(lambda x : x.replace("BPXSY", ""))
dx = dx.sort_values(by="SEQN")

dx["RIAGENDRx"] = dx.RIAGENDR.replace({1: "Male", 2: "Female"})

dx["RIDRETH1x"] = dx.RIDRETH1.replace({1: "MexicanAmerican", 2: "OtherHispanic",
                   3: "NonHispanicWhite", 4: "NonHispanicBlack", 5: "Other"})
# -

# Next we fit a GEE with linear mean structure to the systolic blood
# pressure values (two observations per subject).  The exchangeable
# correlation structure allows us to estimate the ICC (i.e. how
# correlated are the two repeated observations within a subject).
# Note that we are now ignoring the correlations induced by the survey
# structure.

# +
dx["RIDAGEYRm18"] = dx.RIDAGEYR - 18

model = sm.GEE.from_formula("SBP ~ RIDAGEYRm18 * RIAGENDRx + RIDRETH1x + Time",
                            cov_struct=sm.cov_struct.Exchangeable(),
                            groups="SEQN", data=dx)
result = model.fit()
print(result.summary())
print(result.cov_struct.summary())
# -

# It is also possible to have multiple levels of grouping structure in
# a GEE.  Here, we have two repeated measures clustered within each
# individual, and the individuals are in turn clustered within the
# survey groups.  We can use a `Nested` dependence structure to
# capture both levels of correlation.

# +
dx["group"] = 10 * dx.SDMVSTRA + dx.SDMVPSU

model = sm.GEE.from_formula("SBP ~ RIDAGEYRm18 * RIAGENDRx + RIDRETH1x",
                            dep_data="0 + SEQN",
                            cov_struct=sm.cov_struct.Nested(),
                            groups="group", data=dx)
result = model.fit()
print(result.summary())
print(result.cov_struct.summary())
# -

# The fitted nested covariance structure is expressed in terms of the
# variance contributed by each level of nesting.  This corresponds to
# a "variance components model" of the form

# $$
# y_{ijkl} = a_i + b_{ij} + c_{ijk} + \epsilon_{ijkl}
# $$

# Above, $a_i$, $b_{ij}$, $c_{ijk}$, and $\epsilon_{ijkl}$ are all
# random with mean zero, and with variances $\tau_a^2$, $\tau_b^2$,
# $\tau_c^2$, and $\sigma^2$ respectively.

# ## Simulation of data for GEE analysis

# Sometimes it is useful to apply a regression approach to simulated
# data.  This can be applied in power and sample size assessment, and
# in evaluating the performance of a statistical methodology.
# Simulation of data for GEE regression presents a unique challenge
# because GEE is not based on a complete model for the data.  In a
# GEE, the mean structure is fully specified, and in most cases the
# marginal distribution of each observation is fully specified
# (however this is not true when using the quasi-likelihood familes).
# The dependence structure is only specified through a working
# structure that does not need to be correct, and even if it is
# correct, it does not define the joint distribution of the data
# (except in very special cases like the Gaussian model).

# There are various ways to resolve this challenge.  Below we will
# demonstrate how a copula approach can be used to simulate data for a
# GEE analysis.  The basic idea is that if $u_1, u_2$ are two random
# variables that may be dependent, such that $u_1$ and $u_2$ are both
# marginally uniformly distributed, then for any cumulative
# distribution functions $F_1$, $F_2$, the random variables $y_1 =
# F_1^{-1}(u_1)$ and $y_2 = F_2^{-1}(u_2)$ are distributed according
# to $F_1$ and $F_2$.  In a GEE, we know the marginal distributions
# (except in the quasi-likelihood case that we will not cover here),
# so $F_1$, etc. are always known.  Thus, we can induce dependence
# between $y_1$ and $y_2$ by simulating $u_1$ and $u_2$ to be
# dependent and (marginally) uniform.  Note that this approach works
# for any dimension, we are only using two components here for
# illustration.

# One way to simulate dependent $u_j$ that are marginally uniform is
# to simulate dependent $z_j$ that are marginally standard Gaussian,
# then define $u_j = \Phi^{-1}(z_j)$, where $\Phi$ is the standard
# normal CDF.  This approach is flexible because it is easy to
# generate Gaussian random vectors with a variety of different
# covariance structures, such that the components are marginally
# standard normal. This is called the "Gaussian copula" and is used in
# the illustration below.

# +
# Number of groups
n = 1000

# Number of observations per group
q = 10

# The autocorrelation in the Gaussian copula
a = 0.7

from scipy.stats.distributions import poisson, norm

# The latent variables in the Gaussian copula
z = np.random.normal(size=(n, q))
for j in range(1, q):
    z[:, j] = a*z[:, j-1] + np.sqrt(1 - a**2)*z[:, j]
u = norm.cdf(z)

# The covariates
x1 = np.random.normal(size=(n, q))
x2 = np.random.normal(size=(n, q))

# The mean parameters for the marginal distributions
lpr = x1 - 0.5*x2
expval = np.exp(lpr)

# The response values.  These are marginally Poisson with the specified means.
y = np.zeros((n, q))
for i in range(n):
    for j in range(q):
        y[i, j] = poisson.ppf(u[i, j], expval[i, j])

idv = np.outer(np.arange(n), np.ones(q))
time = np.outer(np.ones(n), np.arange(q))

df = pd.DataFrame({"y": y.flat, "x1": x1.flat, "x2": x2.flat,
                   "grp": idv.flat, "time":time.flat})

model = sm.GEE.from_formula("y ~ x1 + x2", groups="grp",
           family=sm.families.Poisson(),
           time=df.time,
           cov_struct=sm.cov_struct.Stationary(max_lag=5),
           data=df)
result = model.fit()

print(result.summary())
print(result.cov_struct.summary())
# -

# ## Other applications of GEE

# Here we have shown how GEE can be used with grouped data, including
# multilevel (nested) grouped data.  We have also seen an application
# involving data that were simulated as a time series.  GEE is widely
# used for grouped data, longitudinal data, time series, and spatial
# data.  In principle, covariance structures can be defined for most
# applications involving dependent data.  However in practice most
# people use the familiar dependence structures illustrated here.

# Finally, note that most of the GEE support within Python Statsmodels
# is implemented in [this source
# file](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/genmod/generalized_estimating_equations.py).
# You do not need to read or understand this source code to be able to
# use GEE regression in Statsmodels.  But more advanced users may want
# to understand how the fitting is implemented.  GEE constitutes a
# fairly mature and well-established methodology.  Therefore, the
# results of fitting a GEE to a data set should be essentially
# identical in all packages (Python Statsmodels, R, Stata, SAS, etc.).
# Statsmodels uses a large number of unit tests
# (e.g. [here](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/genmod/tests/test_gee.py))
# to ensure that the results are correct and consistent with other
# packages.
