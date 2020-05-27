# ## SARS-CoV-2 testing in relation to Covid mortality

# This notebook considers several topics related to Covid mortality
# and testing, using data from the [Covid Tracking
# project](covidtracking.com).
#
# You can get the data by going to https://covidtracking.com/api, then
# downloading the CSV file under "historic state data".  Make sure to
# save the data using file name "ctp_daily.csv".  A direct link to the
# data is here:
#
# https://covidtracking.com/api/v1/states/daily.csv
#
# We will also use population data for the US states.  This can be
# obtained from various sources, we will get it directly from
# Wikipedia below.
#
# Our main interest here will be to assess whether the reported
# testing data for active viral infection, using the PCR testing
# platform, are sufficient to explain the variation in Covid-related
# mortality, all using daily-level data.  There are other possible
# uses of testing such as for use in contact tracing, identifying
# emerging hotspots, and assessing cumulative prevalence of past
# exposure to the virus.  These topics are not considered further
# here.

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# Below we get state population data from Wikipedia.  We only download
# the file one time and store it locally as 'states.csv'.
if not os.path.exists("states.csv"):

    # Map full state names to postal codes
    dm = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
        "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
        "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
        "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
        "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
        "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
        "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
        "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Puerto Rico": "PR", "Rhode Island": "RI",
        "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
        "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI",
        "Wyoming": "WY"}

    ta = pd.read_html("https://simple.m.wikipedia.org/wiki/List_of_U.S._states_by_population",
                      attrs={"class": "wikitable sortable"})
    ta = ta[0].iloc[:, 2:4]
    ta["SA"] = [dm.get(x, "") for x in ta.State]
    ta = ta.loc[ta.SA != "", :]
    ta.columns = ["state", "pop", "SA"]
    ta = ta.loc[:, ["state", "SA", "pop"]]
    ta = ta.sort_values(by="state")
    ta.to_csv("states.csv", index=None)

# Load and merge the Covid data and the state population data.

df = pd.read_csv("ctp_daily.csv")
dp = pd.read_csv("states.csv")
dp = dp.drop("state", axis=1)
df = pd.merge(df, dp, left_on="state", right_on="SA", how="left")
dx = df.loc[:, ["date", "state", "positive", "negative", "death", "pop"]].dropna()
dx = dx.sort_values(by=["state", "date"])

# The data as provided are cumulative, so we difference it to get the
# daily values.

dx["ddeath"] = dx.groupby("state")["death"].diff()
dx["dpositive"] = dx.groupby("state")["positive"].diff()
dx["dnegative"] = dx.groupby("state")["negative"].diff()

# ## Relationship between testing results and mortality.
#
# The Covid Tracking project reports daily counts of Covid-related
# deaths for each US state, and also reports the number of positive
# and negative Covid tests.  The tests reported are primarily PCR
# tests that assess for for viral RNA, which should indicate an active
# or recently cleared SARS-CoV-2 infection.
#
# There is a lot of debate and discussion about the properties of PCR
# testing for SARS-CoV-2.  One useful reference for this topic is:
#
# https://asm.org/Articles/2020/April/False-Negatives-and-Reinfections-the-Challenges-of
#
# To summarize:
#
# * It is currently unknown for how long after infection, symptom
# onset, or recovery a person may continue to have positive PCR test
# values.  It is likely that this duration could be from 1 to 4 weeks
# in most cases.
#
# * The PCR test is quite sensitive, approaching 100% in lab
# conditions, but false negatives can result from flawed sample
# collection.
#
# * The PCR test is quite specific in the narrow sense -- if it
# detects viral RNA, then viral RNA was probably present.  However it
# is possible that PCR may detect residual non-pathogenic RNA remnants
# from a cleared infection.
#
# * Viral load is highest early in the course of the infection,
# including during the pre-symptomatic or asymptomatic period. People
# with severe Covid complications that persist several weeks after
# symptoms appear may have minimal viral load by that time, and are
# primarily impacted by the after-effects of their immune response to
# the virus.
#
# * It is currently unknown what fraction of the many people with
# minimal or no symptoms following SARS-CoV-2 exposure will test
# positive via PCR.
#
# * Antibody testing detects past infection, which in most cases will
# have resolved to the point where a concurrent PCR test would be
# negative.  We only have basic summaries of the antibody testing that
# has been done to date.  Reportedly, a few states have counted
# antibody tests among the PCR tests.  We do not attempt to account
# for that here.
#
# We should also keep in mind that each US state follows different
# practices for testing and reporting.  For mortality data, there may
# be differences in which deaths are deemed to be Covid-associated.
# There are likely to be substantial undercounts, for example some
# states only reported deaths in hospitals.  But there could be some
# overcounting as well, e.g. a person with multiple severe illnesses,
# including Covid, may not have died primarily due to Covid.
#
# For the testing data, each state has its own policies about who can
# get tested.  Early in the epidemic testing was severely constrained
# by the availability of test kits.  Not all states have consistently
# reported negative test results, but reporting of test positives is
# presumably somewhat more consistent.  The sample of tested people is
# not representative of the general population, and is certainly
# enriched for cases.

# Our first analysis will use a regression approach to relate
# mortality to the number of people exposed to the virus.  Since
# people who die from Covid typically were infected several weeks
# prior to their death, we will create counts of testing positives and
# negatives in several week-long windows lagging behind the mortality
# count.

# +
# Sum x from d2 days back in time to d1 days back in time, inclusive of
# both endpoints.  d2 must be greater than d1.
def wsum(x, d1, d2):
    w = np.ones(d2 + 1)
    if d1 > 0:
        w[-d1:] = 0
    y = np.zeros_like(x)
    y[d2:] = np.convolve(x.values, w[::-1], mode='valid')
    return y

for j in range(4):
    dx["cumpos%d" % j] = dx.groupby("state").dpositive.transform(lambda x: wsum(x, 7*j, 7*j+6))
    dx["cumneg%d" % j] = dx.groupby("state").dnegative.transform(lambda x: wsum(x, 7*j, 7*j+6))
    dx["logcumpos%d" % j] = np.log(dx["cumpos%d" % j] + 1)
    dx["logcumneg%d" % j] = np.log(dx["cumneg%d" % j] + 1)
# -

# Below we will use regression analysis to try to understand how
# SARS-CoV-2 testing results relate to Covid mortality.  A reasonable
# hypothesis would be that a greater number of positive test results
# predicts greater mortality, at a 1-3 week lag.  More specifically,
# all else held equal, comparing two places/times where the Covid
# prevalence differs by a factor of two, we might expect there to be a
# factor of two difference in Covid mortality at some later date.
#
# Another reasonable hypothesis would be that the ratio of positive to
# negative test results, rather than the absolute number of positive
# test results, would be a stronger predictor of mortality.  This is
# based on the logic that taking this ratio corrects for the fact that
# states may increase their testing when a big surge in cases is
# expected.
#
# There are several other factors that we should consider and account
# for if possible:
#
# * The virus arrived in different states at different times, e.g. it
# arrived in New York before it arrived in South Dakota.
#
# * States vary greatly in terms of population size.  It is reasonable
# to expect death counts to scale with population size.
#
# * Transmission rates may vary by state, e.g. due to population
# density.
#
# * The seasons were changing just as the epidemic in the US reached
# its peak.  Spring weather may reduce transmission rates.
#
# * The infection/fatality ratio (IFR) may vary by state due to
# demographic characteristics of the population and prevalence of
# comorbidities.

# To account for differences in the time when the disease arrived in
# each state, we identify the date of the first Covid death, then
# exclude the 10 days following that date.

# +
def firstdeath(x):
    if (x.death == 0).all():
        return np.inf
    ii = np.flatnonzero(x.death > 0)[0]
    return x.date.iloc[ii]

xx = dx.groupby("state").apply(firstdeath)
xx.name = "firstdeath"
dx = pd.merge(dx, xx, left_on="state", right_index=True)

dx["rdate"] = dx.date - dx.firstdeath
dx = dx.loc[dx.rdate >= 10, :]
# -

# Below is an initial regression analysis looking at mortality as a an
# outcome that is predicted by testing results.  We include state
# level fixed effects to control for different infection/fatality
# ratios among the states.  The model is fit using a Poisson
# generalized estimating equations (GEE) approach, clustering the data
# by state.  This allows data values in the same state to be
# correlated, e.g. through a serial correlation with respect to time.

fml = "ddeath ~ 0 + C(state) + "
fml += " + ".join(["logcumpos%d" % j for j in range(4)])
fml += " + "
fml += " + ".join(["logcumneg%d" % j for j in range(4)])
m1 = sm.GEE.from_formula(fml, groups="state", data=dx, family=sm.families.Poisson())
r1 = m1.fit(scale="X2")
print(r1.summary())

# A perfect 1-1 relationship between testing and mortality would
# manifest as a perfect linear relationship between the log number of
# cases and the log number of positive tests.  In this case, we would
# expect the regression coefficients for the log number of positive
# tests to be equal to 1 (assuming that cases and mortality are 1-1 at
# some lag).  As we see above, the coefficient for the log number of
# positive tests in the week immediately preceeding the mortality
# count is around 0.56, and the coefficients for the two weeks prior
# to that were 0.19 and 0.04 respectively.  Note that these
# coefficients sum to around 0.8.  This is consistent with an
# interpretation in which the number of positive tests captures around
# 80% of the infections -- if the positive tests in two states/4 week
# time periods differ by a factor of two, the state with more positive
# tests will have around 80% more deaths.  The fact that these effects
# are spread across several lagged terms may reflect variation in the
# timing of death relative to infection.  We can argue that the
# remaining 20% of deaths were missed due to inadequate testing.
#
# Note also that there is a statistically significant negative
# relationship between the log number of negative tests in the prior
# weeks and mortality.  However the coefficients of the negative test
# result variables do not sum to a value close to 1.  This would argue
# against using the ratio of the number of positive to negative tests
# as a predictor of mortality.

# ## State effects and variation in the IFR

# If we believe that the positive and negative testing data are
# sufficiently informative about the prevalence of the disease at each
# state/time, then the state fixed effects in the above model might
# reflect state-to-state differences in the infection/fatality ratio.
# The analysis below shows that these state effects have a range of
# around 5.26 on the log scale, with a standard deviation of 0.86.
# Since exp(0.86) ~ 2.36, this might suggest that most states have IFR
# values that are within a factor of 2.5 of the mean state-level IFR
# -- however, see below for a more refined analysis.

# +
# Extract the state fixed effects
pa = r1.params
st = [y for x, y in zip(pa.index, pa.values) if "state" in x]
st = np.asarray(st)

print("Range of state effects:")
print(st.min(), st.max())

print("Unadjusted SD of state effects:")
print(st.std())
# -

# The state fixed effects discussed above are estimates, not exact
# values.  The estimation errors in these quantities inflate their
# variance when estimated naively as in the preceding cell.  This can
# be addressed as shown below.  The results suggest that most states
# are within 12% of the mean state, suggesting that the IFR values may
# be quite similar from state to state.

# +
ii = [i for i, x in enumerate(pa.index) if "state" in x]
c = r1.cov_params().iloc[ii, ii]

# The centering matrix
p = len(st)
oo = np.ones(p)
qm = np.eye(p) - np.outer(oo, oo) / p

vv = np.trace(np.dot(qm, np.dot(c, qm))) / p
print("Adjusted variance and SD of state effects:")
print("variance=%f, SD=%f\n" % (vv, np.sqrt(vv)))
# -

# As with any observational data set, there are many opportunities for
# confounding to mislead us.  One such possibility is that all the US
# states have progressed through the Covid epidemic at roughly the
# same time period, and during this time the weather became much
# warmer in most of the US.  In fact, weather is just one possible
# confounder indexed by time.  To address this possibility, we fit
# models in which calendar date, or date relative to the state's first
# Covid death are included as controls.  We find that the coefficients
# for the positive and negative testing results are relatively invariant to
# inclusion of these terms.

fml = "ddeath ~ C(state) + bs(rdate, 5) + "
fml += " + ".join(["logcumpos%d" % j for j in range(4)])
fml += " + "
fml += " + ".join(["logcumneg%d" % j for j in range(4)])
m2 = sm.GEE.from_formula(fml, groups="state", data=dx, family=sm.families.Poisson())
r2 = m2.fit(scale="X2")
print(r2.summary())
print("Score test for model 2 relative to model 1:")
print(m2.compare_score_test(r1))

# As noted above, it makes sense to model death relative to population
# size.  Since the Poisson regression that we will be using has a log
# link function, to properly account for population effects we should
# use the log of the total population as an offset, or as a covariate.
# Note however that Covid has largely arisen in geographically-limited
# clusters that are smaller than an entire state.  For this reason,
# the state population may not be the perfect offset to use for this
# purpose.  Ideally, we would have mortality and testing data at a
# finer geographical scale, for example by county.  But that data is
# not available now.

dx["lpop"] = np.log(dx["pop"])

# Since population size is a state-level variable and we already have
# included state fixed effects in the model, a main effect of
# population size has already been accounted for in the models above.
# However we can take the question of population scaling a bit further
# by considering interactions between population size and the positive
# test counts.  As shown below however, these coefficients are not
# statistically significant based on the score test.

dx["lpop_cen"] = dx.lpop - dx.lpop.mean()
fml = "ddeath ~ C(state) + bs(rdate, 5) + "
fml += " + ".join(["lpop_cen*logcumpos%d" % j for j in range(4)])
fml += " + "
fml += " + ".join(["lpop_cen*logcumneg%d" % j for j in range(4)])
m3 = sm.GEE.from_formula(fml, groups="state", data=dx, family=sm.families.Poisson())
r3 = m3.fit(scale="X2")
print("Score test for model 3 relative to model 2:")
print(m3.compare_score_test(r2))

# ## Dispersion relative to the mean and the scale parameter

# Above we focused on the mean structure of the model, which is
# reflected in the slope parameters for the covariates.  These
# parameters determine the expected value for any given state/date
# pair.  We should also consider to what extent the data are scattered
# with respect to this mean.  This is captured through the scale
# parameter of the quasi-Poisson regression.

print(r1.scale)
print(r2.scale)
print(r3.scale)

# Focusing on the scale parameters for model 2 above, which have
# values around 10, we see that the variance of our data is around 10
# times greater than the mean.  In a perfect Poisson situation, the
# variance would be equal to the mean.  This perfect Poisson behavior
# would arise if we had independence and homogeneity -- independence
# meaning that any two people living in the same state on the same day
# die of Covid independently of each other, and homogeneity meaning
# that any two people living in the same state on the same day have
# the same probability of dying of Covid.  The independence condition
# is likely to hold, but the homogeneity condition is not.  Our data
# are not stratified by known risk factors, sex and age being most
# well-established.  Pooling data with different event probabilities
# would give us a scale parameter greater than 1, as is seen here.
#
# Although we don't have access to the stratified data that we would
# need to definitively identify the sources of dispersion, we can do
# some sensitivity analyses to see to what extent age and sex effects
# might inflate the scale parameter.

# The following function implements a form of sensitivity analysis in
# which we attempt to identify population structures that are
# conistent with the findings we reached above with the Covid tracking
# project data.  Our goal here is to get a better sense for how much
# heterogeneity would be needed to attain a scale parameter around 18,
# while respecting the marginal mean structure estimated above.  We
# can do this, in a very hypothetical way, by imagining that our
# population consists of two groups with different risks.  If we
# specify the prevalences of these two groups, and the risk ratio
# between them, and force the marginal mean to agree with what we
# found above, we can calculate the scale parameter.
#
# This idea is implemented in the function below.  The parameter
# 'high_risk' is a hypothetical risk ratio between a high risk group
# and a low risk group.  The parameter 'pr_high' is the hypothetical
# proportion of the population in the high risk group.

def scale_2group(high_risk, pr_high):
    f = np.r_[1, high_risk]
    pr = np.r_[1 - pr_high, pr_high]
    f /= np.dot(f, pr)
    ex = r1.fittedvalues
    exq = [fx*ex for fx in f]
    mcv = sum([p*b for (p, b) in zip(pr, exq)])
    vcm = sum([p*(b-ex)**2 for (p, b) in zip(pr, exq)])
    tv = mcv + vcm
    return (tv/ex).mean()

# If we calculate the hypothetical scale parameter for an imaginary
# population in which 5% of the population is 4 times more likely to
# die than the others, we get a scale parameter of around 11, which is
# close to our observed scale parameter.  Of course the real world
# does not consist of two homogeneous groups, but this may not be as
# far as one would imagine from reality.  For example, including a
# third group with very low risk would not change anything.
# Consistent with observed data, suppose that the risk of dying for
# people under 40 is negligible.  Then the scenario we are considering
# posits that among people over 40, 95% of them have a certain base
# risk, and the remainder have a risk that is elevated by a factor of
# 4.

scale_2group(4.0, 0.05)

# ## Infection fatality ratio

# The infection fatality ratio is defined to be the ratio between the
# number of people who die of Covid and the number of people exposed
# to SARS-CoV-2.  This ratio could in principle change throughout the
# progress of the epidemic, and could differ in various locations.
# However it is likely to primarily depend on relatively tangible and
# stable demographic characteristics, and characteristics relating to
# baseline health risk.
#
# With caveats as noted above, we have established to some extent that
# positive PCR test results and Covid mortality covary directly.  This
# may imply that positive PCR test results covary directly with
# SARS-CoV-2 exposure, and that this in turn covaries directly with
# Covid.  We will not be able to establish this definitevly, since
# there is a lot of uncertainty about who has been exposed.
#
# However as a form of sensitivity analysis, we can use side
# information to assess how our observables (mortality and PCR tests)
# are related to the key unobservable, which is the cumulative number
# of people exposed in each state, at each point in time.
#
# The key concept here is the "case ascertainment ratio", which is the
# number of people exposed to the virus for each person who tests
# positive.  Informed speculation about this number places it between
# 5 and 50, although values from 2 to 100 have been claimed.
