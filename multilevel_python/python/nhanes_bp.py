# Fit mixed models to NHANES blood pressure data.  There are two blood
# pressure measurement types (systolic and diastolic), with up to 4
# repeated measures for each type.
#
# https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT
# https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/BPX_G.XPT
# https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/BMX_G.XPT

import statsmodels.api as sm
import pandas as pd
import numpy as np

# Load and merge the data sets

demog = pd.read_sas("../data/DEMO_G.XPT")
bpx = pd.read_sas("../data/BPX_G.XPT")
bmx = pd.read_sas("../data/BMX_G.XPT")
df = pd.merge(demog, bpx, left_on="SEQN", right_on="SEQN")
df = pd.merge(df, bmx, left_on="SEQN", right_on="SEQN")

# Convert from wide to long

syvars = ["BPXSY%d" % j for j in (1,2,3,4)]
divars = ["BPXDI%d" % j for j in (1,2,3,4)]
vvars = syvars + divars
idvars = ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'BMXBMI']
dx = pd.melt(df, id_vars=idvars, value_vars=vvars,
             var_name='bpvar', value_name='bp')

# A bit of data cleanup

# +
dx = dx.sort_values(by='SEQN')
dx = dx.reset_index(drop=True)
dx['SEQN'] = dx.SEQN.astype(np.int)
dx = dx.dropna()

# Blood pressure type (systolic or diastolic)
dx["bpt"] = dx.bpvar.str[3:5]

dx["bpi"] = dx.bpvar.str[5].astype(np.int)
dx["female"] = (dx.RIAGENDR == 2).astype(np.int)

di_mean = dx.loc[dx.bpt=="DI", :].groupby("SEQN")["bp"].aggregate(np.mean)
di_mean.name = "di_mean"
dx = pd.merge(dx, di_mean, left_on="SEQN", right_index=True)

print(dx.head())
# -

# Subsample to make the script run faster

dx = dx.iloc[0:10000, :]

# Fit a linear mean structure model using OLS. The variance structure of
# this model is misspecified.

model1 = sm.OLS.from_formula("bp ~ RIDAGEYR + female + C(bpt) + BMXBMI", dx)
result1 = model1.fit()
print(result1.summary())

# Fit a mixed model to systolic data with a simple random intercept
# per subject.  Then calculate ICC.

ds2 = dx.loc[dx.bpt == "SY"]
model2 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + BMXBMI",
                                 groups="SEQN", data=ds2)
result2 = model2.fit()
icc2 = result2.cov_re / (result2.cov_re + result2.scale)
print(result2.summary())
print("icc=%f\n" % icc2.values.flat[0])

# Partial out the mean diastolic blood pressure per subject

model3 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + BMXBMI + di_mean",
                                 groups="SEQN", data=ds2)
result3 = model3.fit()
icc3 = result3.cov_re / (result3.cov_re + result3.scale)
print(result3.summary())
print("icc=%f\n" % icc3.values.flat[0])

# Fit a mixed model to diastolic data only with simple random
# intercept per subject.  Then calculate ICC.

ds3 = dx.loc[dx.bpt == "DI"]
model4 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + BMXBMI",
                                 groups="SEQN", data=ds3)
result4 = model4.fit()
icc4 = result4.cov_re / (result4.cov_re + result4.scale)
print(result4.summary())
print("icc=%f\n" % icc4.values.flat[0])

# Fit a mixed model to diastolic data only with simple random
# intercept per subject (also using subset of data).

ds3 = dx.loc[dx.bpt == "DI"]
model5 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + BMXBMI + bpi",
                                 groups="SEQN", re_formula="1+bpi",
                                 data=ds3)
result5 = model5.fit()
print(result5.summary())

# Fit a mixed model to both types of BP with simple random intercept
# per subject.

model6 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + C(bpt) + BMXBMI",
                                 groups="SEQN", data=dx)
result6 = model6.fit()
print(result6.summary())

# Fit a mixed model to both types of BP with subject random intercept
# and unique random effect per BP type with common variance.

model7 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + C(bpt) + BMXBMI",
                                 groups="SEQN", re_formula="1",
                                 vc_formula={"bpt": "0+C(bpt)"},
                                 data=dx)
result7 = model7.fit()
print(result7.summary())

# Fit a mixed model to both types of BP with subject random intercept
# and unique random effect per BP type with unique variance.

dx["sy"] = (dx.bpt == "SY").astype(np.int)
dx["di"] = (dx.bpt == "DI").astype(np.int)
model8 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + C(bpt) + BMXBMI",
                                 groups="SEQN", re_formula="1",
                                 vc_formula={"sy": "0+sy", "di": "0+di"},
                                 data=dx)
result8 = model8.fit()
print(result8.summary())

# Fit a mixed model to both types of BP with subject random intercept
# and unique random effect per BP type with unique variance, and
# heteroscedasticity by BP type.

dx["sy1"] = (dx.bpvar == "BPXSY1").astype(np.int)
dx["sy2"] = (dx.bpvar == "BPXSY2").astype(np.int)
dx["sy3"] = (dx.bpvar == "BPXSY3").astype(np.int)
dx["di1"] = (dx.bpvar == "BPXDI1").astype(np.int)
dx["di2"] = (dx.bpvar == "BPXDI2").astype(np.int)
dx["di3"] = (dx.bpvar == "BPXDI3").astype(np.int)
model9 = sm.MixedLM.from_formula("bp ~ RIDAGEYR + female + C(bpt) + BMXBMI",
                                 groups="SEQN", re_formula="1",
                                 vc_formula={"sy": "0+sy", "di": "0+di",
                                             "dye": "0+di1+di2+di3"},
                                 data=dx)
result9 = model9.fit()
print(result9.summary())
