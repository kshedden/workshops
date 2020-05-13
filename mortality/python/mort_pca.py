# ## Generalized Principal Component Analysis of US Mortality Data

# This notebook uses generalized Principal Component Analysis
# to aid in understanding the patterns of variation in US
# mortality data.

# These are the modules that we will be using here:

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from gpca import GPCA

# This is the path to the data file on your system.  You will need
# to change this to match the path used in the `vital_stats_prep.py`
# script.

pa = "/nfs/kshedden/cdc_mortality/final/pop_mort.csv"
df = pd.read_csv(pa)

# PCA operates on a rectangular data matrix.  Depending on what
# we want to focus on, we can choose one of several ways to
# create this matrix.  Here, we will pivot the age groups to
# the columns to create a rectangular matrix.

ix = ["Year", "Month", "Sex", "Age_group"]
dx = df.set_index(ix)
deaths = dx.loc[:, "Deaths"].unstack("Age_group")
deaths = deaths.fillna(1)

# We may wish to adjust for certain known factors, so that the PCA
# focuses on the variation around these factors.  A basic
# factor to adjust for is the population size, so that we are
# looking at mortality with respect to population size ("death rates").
pop = dx.loc[:, "Population"].unstack("Age_group")
lpop = np.log(pop)

# Below we fit the generalized PCA, using a Poisson family and
# two factors.

pca = GPCA(deaths, 2, offset=lpop, family=sm.families.Poisson())
r = pca.fit(maxiter=2000)

# The warning about convergence may be ignored as the norm of
# the gradient is rather small.

r.score_norm

# A generalized PCA identifies "intercepts", analogous to means,
# such that variables with lesser intercepts tend to have smaller
# values than variables with greater intercepts. The plot below shows
# the intercepts.

# +
ages = deaths.columns
plt.clf()
plt.grid(True)
plt.plot(ages, r.intercept)
plt.xlabel("Ages")
plt.ylabel("Intercepts")
xl = plt.gca().xaxis.get_ticklabels()
for x in xl:
    x.set_rotation(90)
# -

# The intercepts are usually of lesser interest than the factors,
# which are analogous to traditional "principal components".  Below
# we plot the loadings of the two-factor fit.  Note that the loadings
# for each factor consist of a sequence of values corresponding to the
# variables (the age categories).

# The dominant factor has entirely positive loadings, and the loadings
# are especially high for ages from 20 to 40.  This means
# that it captures a pattern of variation in which demographic/temporal
# cells that score positively on this factor have greater mortality
# for all ages, especially for ages 20-40, and demographic/temporal cells
# that score negatively on this factor have lesser mortality for all ages
# (especially for ages 20-40).  Below we will determine which cells
# score higher or lower on this factor.

# The second factor captures a pattern in which mortalilty is lower
# between ages 20 and 40, and higher for older people (or vice-versa,
# depending on the sign of the score).

# +
plt.grid(True)
plt.plot(ages, r.factors[:, 0], label="1")
plt.plot(ages, r.factors[:, 1], label="2")
ha, lb = plt.gca().get_legend_handles_labels()
plt.figlegend(ha, lb, "center right")
plt.xlabel("Ages")
plt.ylabel("Factor loadings")
xl = plt.gca().xaxis.get_ticklabels()
for x in xl:
    x.set_rotation(90)
# -

# One of the best ways to understand the factors is to look at the scores,
# which correspond to the rows of the data matrix.  The scores in our case
# are records of deaths in sex x month x year cells.

# To make this plot, first prepare a dataframe containing the scores
# together with some other relevant variables.

# +
scores = pca.scores(r.params)

dm = dx.index.to_frame().unstack("Age_group")
sex = dm.loc[:, ("Sex", "70_74")].values
month = dm.loc[:, ("Month", "70_74")].values
year = dm.loc[:, ("Year", "70_74")].values

scores = pd.DataFrame(scores, columns=["factor1", "factor2"])
scores["sex"] = sex
scores["month"] = month
scores["year"] = year
# -

# The following plot showing the two scores plotted against
# each other makes it clear that factor 1 is largely dependent
# on sex -- males score uniformly higher on factor 1 compared
# to females. Recall that factor 1 corresponds to greater
# mortality in all age bands, especially ages 20-40.
#
# Factor 2 is also dependent on sex, but to a much lesser
# degree than factor 1 (there is a small tendency for the male
# cells to score higher on factor 2 than the female cells).

# +
plt.grid(True)
col = {"Female": "purple", "Male": "orange"}
for s in "Female", "Male":
    ii = np.flatnonzero(scores.sex == s)
    plt.plot(scores.factor1[ii], scores.factor2[ii], 'o', color=col[s], label=s)
ha, lb = plt.gca().get_legend_handles_labels()
leg = plt.figlegend(ha, lb, "upper center", ncol=2)
leg.draw_frame(False)
plt.xlabel("Scores for component 1")
plt.ylabel("Scores for component 2")
# -

# Next we will look at how the factor scores vary by month.
# We first create a function that will generate a scatterplot
# of the factor scores against the month, for any factor
# in the model.

# +
def plot_month(factor):
    plt.clf()
    plt.grid(True)
    for sex in "Female", "Male":
        scores1 = scores.loc[scores.sex == sex, :]
        plt.plot(scores1.month, scores1.loc[:, factor], 'o', label=sex)
    plt.xlabel("Month")
    plt.ylabel("Score")
    plt.title(factor)
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)
# -

# This is the scatterplot of factor 1 scores against month.  This is
# dominated by the sex effect, with little evidence of a month effect.

# +
plot_month("factor1")
# -

# Next we create a scatterplot of factor 2 scores against month.  It
# shows clearly that summer months tend to have negative scores for
# factor 2, and winter months tend to have positive scores for factor
# 2.  This indicates that winter months have greater mortality among
# the elderly and leseer mortality among young adults (compared to the
# intercepts), while summer months have somewhat less age-specific
# variation in the mortality patterns.

# +
plot_month("factor2")
# -

# +
def plot_year(factor):
    plt.clf()
    plt.grid(True)
    for sex in "Female", "Male":
        scores1 = scores.loc[scores.sex == sex, :]
        plt.plot(scores1.year, scores1.loc[:, factor], 'o', label=sex)
    plt.xlabel("Year")
    plt.ylabel("Score")
    plt.title(factor)
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.draw_frame(False)
# -

# Next we look at how the factor scores vary by year.  The factor 1
# scores are slightly higher at the beginning and end of the time
# range covered by these data, with a minimum around 2012.  This
# may reflect a steady decline in mortality that reversed for
# a few years after 2012.

# +
plot_year("factor1")
# -

# The factor 2 scores decrease over the 12 years of data considered
# here.  This seems to indicate a weakening of the tendency for mortality
# to be concentrated among older people in the winter months.

# +
plot_year("factor2")
# -
