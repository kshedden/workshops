# # Multilevel analysis of NHANES blood pressure data

# In this notebook, we fit mixed models to blood pressure data from NHANES.
# The study has data for two blood
# pressure measurement types (systolic BP and diastolic BP), with up to 4
# repeated measures per subject for each type.
#
# https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT
# https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/BPX_G.XPT
# https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/BMX_G.XPT

import statsmodels.api as sm
import pandas as pd
import numpy as np

# First, load and merge the data sets.  These are SAS Xport format
# files, which can be read with Pandas.

demog = pd.read_sas("../data/DEMO_G.XPT")
bpx = pd.read_sas("../data/BPX_G.XPT")
bmx = pd.read_sas("../data/BMX_G.XPT")
df = pd.merge(demog, bpx, left_on="SEQN", right_on="SEQN")
df = pd.merge(df, bmx, left_on="SEQN", right_on="SEQN")

# Next we convert the data from wide to long, pivoting the four
# BP measures from columns to rows.

syvars = ["BPXSY%d" % j for j in (1,2,3,4)]
divars = ["BPXDI%d" % j for j in (1,2,3,4)]
vvars = syvars + divars
idvars = ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'BMXBMI']
dx = pd.melt(df, id_vars=idvars, value_vars=vvars,
             var_name='bpvar', value_name='bp')

# We drop rows where any of the variables are missing.  Multilevel modeling
# can accommodate missing data, but here we use a basic complete case analysis.

# +
dx = dx.sort_values(by='SEQN')
dx = dx.reset_index(drop=True)
dx['SEQN'] = dx.SEQN.astype(np.int)
dx = dx.dropna()
# -

# Since we have pivoted all BP measures to rows, we will need variables
# telling us whether we are looking at systolic (SY) or diastolic (DI)
# blood pressure, and we need a way to to know the order of the BP values
# within each person.  These repeated measures are not exchangeable,
# since the BP readings tend to drop slightly as people relax.

# +
# Blood pressure type (systolic or diastolic)
dx["bpt"] = dx.bpvar.str[3:5]

dx["bpi"] = dx.bpvar.str[5].astype(np.int)
dx["female"] = (dx.RIAGENDR == 2).astype(np.int)

di_mean = dx.loc[dx.bpt=="DI", :].groupby("SEQN")["bp"].aggregate(np.mean)
di_mean.name = "di_mean"
dx = pd.merge(dx, di_mean, left_on="SEQN", right_index=True)

print(dx.head())
# -

# Subsample to make the script run faster.  Statsmodels MixedLM is
# unfortunatley not very fast.

dx = dx.iloc[0:5000, :]

# Fit a linear mean structure model using OLS. The variance structure of
# this model is misspecified.  Since this is a linear model, the coefficient
# point estimates are still meaningful despite the variance misspecification.

model1 = sm.OLS.from_formula("bp ~ RIDAGEYR + female + C(bpt) + BMXBMI", dx)
result1 = model1.fit()
print(result1.summary())

# Fit a mixed model to the systolic data with a simple random intercept
# per subject.  Then calculate the ICC.

ds2 = dx.loc[dx.bpt == "SY"]
model2 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + BMXBMI",
                                 groups="SEQN", data=ds2)
result2 = model2.fit()
icc2 = result2.cov_re / (result2.cov_re + result2.scale)
print(result2.summary())
print("icc=%f\n" % icc2.values.flat[0])

# Partial out the mean diastolic blood pressure per subject.  This leads
# to slightly weaker random effects.

model3 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + BMXBMI + di_mean",
                                 groups="SEQN", data=ds2)
result3 = model3.fit()
icc3 = result3.cov_re / (result3.cov_re + result3.scale)
print(result3.summary())
print("icc=%f\n" % icc3.values.flat[0])

# Fit a mixed model to the diastolic data with a random
# intercept per subject.  Then calculate the ICC.

ds3 = dx.loc[dx.bpt == "DI"]
model4 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + BMXBMI",
                                 groups="SEQN", data=ds3)
result4 = model4.fit()
icc4 = result4.cov_re / (result4.cov_re + result4.scale)
print(result4.summary())
print("icc=%f\n" % icc4.values.flat[0])

# Fit a mixed model to the diastolic data with a random
# intercept per subject.

ds3 = dx.loc[dx.bpt == "DI"]
model5 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + BMXBMI + bpi",
                                 groups="SEQN", re_formula="1+bpi",
                                 data=ds3)
result5 = model5.fit()
print(result5.summary())

# Fit the same model as above, now centering the bpi (index) variable.

ds3.loc[:, "bpi_cen"] = ds3.loc[:, "bpi"] - 1
model6 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + BMXBMI + bpi",
                                 groups="SEQN", re_formula="1+bpi_cen",
                                 data=ds3)
result6 = model6.fit()
print(result6.summary())

# Fit a mixed model to both types of BP jointly with a random intercept
# per subject.  Note that the random intercept is share between the two
# types of blood pressure (systolic and diastolic).

model7 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + C(bpt) + BMXBMI",
                                 groups="SEQN", data=dx)
result7 = model7.fit()
print(result7.summary())

# Fit a mixed model to both types of BP with subject random intercept
# and unique random effect per BP type with common variance.

model8 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + C(bpt) + BMXBMI",
                                 groups="SEQN", re_formula="1",
                                 vc_formula={"bpt": "0+C(bpt)"},
                                 data=dx)
result8 = model8.fit()
print(result8.summary())

# Fit a mixed model to both types of BP with subject random intercept
# and unique random effect per BP type with unique variance.

dx["sy"] = (dx.bpt == "SY").astype(np.int)
dx["di"] = (dx.bpt == "DI").astype(np.int)
model9 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + C(bpt) + BMXBMI",
                                 groups="SEQN", re_formula="1",
                                 vc_formula={"sy": "0+sy", "di": "0+di"},
                                 data=dx)
result9 = model9.fit()
print(result9.summary())

# Below we consider the possibility that theremay be heteroscedasticity
# between the two blood pressure types.  That is, systolic blood pressure
# measurements may be more variable than diastolic measurements, or vice
# versa.  This analysis is a bit awkward to conduct.  Below we fit two
# models, one in which diastolic measurements are allowed to be more variable
# than systolic measurements, and one in which systolic measurements are allowed
# to be more variable than diastolic measurements.  In theory, a variance
# parameters should be equal to zero in one of these models, revealing
# which type of blood pressure has more variability.

# +
dx["sy1"] = (dx.bpvar == "BPXSY1").astype(np.int)
dx["sy2"] = (dx.bpvar == "BPXSY2").astype(np.int)
dx["sy3"] = (dx.bpvar == "BPXSY3").astype(np.int)
dx["di1"] = (dx.bpvar == "BPXDI1").astype(np.int)
dx["di2"] = (dx.bpvar == "BPXDI2").astype(np.int)
dx["di3"] = (dx.bpvar == "BPXDI3").astype(np.int)
model10 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + C(bpt) + BMXBMI",
                                 groups="SEQN", re_formula="1",
                                 vc_formula={"sy": "0+sy", "di": "0+di",
                                             "dye": "0+di1+di2+di3"},
                                 data=dx)
result10 = model10.fit()
print(result10.summary())

model11 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + C(bpt) + BMXBMI",
                                 groups="SEQN", re_formula="1",
                                 vc_formula={"sy": "0+sy", "di": "0+di",
                                             "sye": "0+sy1+sy2+sy3"},
                                 data=dx)
result11 = model11.fit()
print(result11.summary())
# -