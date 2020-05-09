## Generalized Principal Component Analysis of US Mortality Data

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

ix = ["Year", "Month", "Sex", "Age_group"]

dx = df.set_index(ix)
deaths = dx.loc[:, "Deaths"].unstack("Age_group")
deaths = deaths.fillna(1)

pop = dx.loc[:, "Population"].unstack("Age_group")
lpop = np.log(pop)

pca = GPCA(deaths, 2, offset=lpop, family=sm.families.Poisson())
r = pca.fit(maxiter=2000)

ages = deaths.columns

# +
plt.grid(True)
plt.plot(ages, r.intercept)
plt.xlabel("Ages")
plt.ylabel("...")
# -

# +
plt.grid(True)
plt.plot(ages, r.factors[:, 0], label="1")
plt.plot(ages, r.factors[:, 1], label="2")
ha, lb = plt.gca().get_legend_handles_labels()
plt.figlegend(ha, lb, "center right")
plt.xlabel("Ages")
plt.ylabel("...")
# -

# +
scores = pca.scores(r.params)
dm = dx.index.to_frame().unstack("Age_group")
sex = dm.loc[:, ("Sex", "70_74")].values

plt.grid(True)
col = {"Female": "purple", "Male": "orange"}
for s in "Female", "Male":
    ii = np.flatnonzero(sex == s)
    plt.plot(scores[ii, 0], scores[ii, 1], 'o', color=col[s], label=s)
ha, lb = plt.gca().get_legend_handles_labels()
leg = plt.figlegend(ha, lb, "center right")
leg.draw_frame(False)
plt.xlabel("Scores for component 1")
plt.ylabel("Scores for component 2")
# -
