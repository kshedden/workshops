# # GEE analysis of growth trajectories of children

# GEE is commonly used in longitudinal data analysis.  Here we
# consider a dataset in which repeated measures of weight were made on
# young children over several years in early childhood.
# GEE allows us to use linear modeling techniques
# similar to OLS, and still rigorously account for the repeated
# measures aspect of the data.

# The data we will use are obtained from this page:
# http://www.bristol.ac.uk/cmm/learning/support/datasets

# These are the packages we will be using:

import pandas as pd
import numpy as np
import statsmodels.api as sm

# The data are in "fixed width" format, so we use some special
# techniques for reading them:

# +
colspecs = [(0, 4), (4, 7), (7, 12), (12, 16), (16, 17)]
df = pd.read_fwf("../data/growth/ASIAN.DAT", colspecs=colspecs, header=None)
df.columns = ["Id", "Age", "Weight", "BWeight", "Gender"]
df["Female"] = 1*(df.Gender == 2)
df = df.dropna()
# -

# Some of the analyses below will use logged data:

# +
df["LogWeight"] = np.log(df.Weight) / np.log(2)
df["LogBWeight"] = np.log(df.BWeight) / np.log(2)
# -

# The first model that we consider treats weight as a linear function of age, and
# ignores the repeated measures structure.  The point estimates from
# this model are valid, but the standard errors are not.

# +
model0 = sm.GLM.from_formula("Weight ~ Age + BWeight + Female", data=df)
rslt0 = model0.fit()
print(rslt0.summary())
# -

# Here is a GEE model with the same mean structure as in the cell
# above, but using GEE gives us meaningful standard errors:

# +
model1 = sm.GEE.from_formula("Weight ~ Age + BWeight + Female", groups="Id", data=df)
rslt1 = model1.fit()
print(rslt1.summary())
# -

# Now we fit the same model as a log/log regression.  Specifically,
# the relationship between weight in childhood at a given age and
# birth weight is modeled as a log/log relationship.  This means that when
# comparing two children of the same sex whose birth weights differed
# by a given percentage, say $x$, then their childhood weights at a
# given age differ on average by a corresponding percentage $b\cdot
# x$, where $b$ is the coefficient of LogBWeight in the model.  Typically we
# anticipate that $0 \le b \le 1$ in this type of regression.  If $b
# \approx 1$ then, say, two kids whose weights at birth differ by
# 20% will continue to have weights differing by 20% as they age.
# If $b < 1$, then the 20% difference at birth will attenuate as the
# kids age.

# +
model2 = sm.GEE.from_formula("LogWeight ~ Age + LogBWeight + Female", groups="Id", data=df)
rslt2 = model2.fit()
print(rslt2.summary())
# -

# It isn't very likely that weight varies either linearly or
# exponentially with age.  We can use splines to capture a much
# broader range of relationships.

# +
model3 = sm.GEE.from_formula("LogWeight ~ bs(Age, 4) + LogBWeight + Female", groups="Id", data=df)
rslt3 = model3.fit()
print(rslt3.summary())
# -

# It is quite possible that the relationships between birth weight and
# childhood weight differ between girls and boys.  An interaction
# captures this possibility.

# +
model4 = sm.GEE.from_formula("LogWeight ~ bs(Age, 4) + LogBWeight*Female", groups="Id", data=df)
rslt4 = model4.fit()
print(rslt4.summary())
# -

# Although GEE does not require us to specify an accurate covariance
# structure, we will have more power if we do so.  We will also learn
# something about the strength of the within-subject dependence that
# we would not learn when using the independence model.

# +
model5 = sm.GEE.from_formula("LogWeight ~ bs(Age, 4) + LogBWeight + Female", groups="Id",
                             cov_struct=sm.cov_struct.Exchangeable(), data=df)
rslt5 = model5.fit()
print(rslt5.summary())
print(rslt5.cov_struct.summary())
# -

# In general, it is better to use the default "robust" approach for
# covariance estimation.  This allows the covariance model to be
# mis-specified, while still yielding valid parameter estimates and
# standard errors.  If you are very confident that your working
# covariance model is correct, you can specify the "naive" approach to
# covariance estimation, as below.  In this case, the standard errors will be
# meaningful only if the working correlation model is correct.

# +
model6 = sm.GEE.from_formula("LogWeight ~ bs(Age, 4) + LogBWeight + Female", groups="Id",
                             cov_struct=sm.cov_struct.Exchangeable(), data=df)
rslt6 = model6.fit(cov_type="naive")
print(rslt6.summary())
# -
