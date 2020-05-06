import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pca_glm import GPCA

# This is the path to the data file on your system.  You will need
# to change this to match the path used in the `vital_stats_prep.py`
# script.

pa = "/nfs/kshedden/cdc_mortality/final/pop_mort.csv"
df = pd.read_csv(pa)

ix = ["Year", "Month", "Sex", "DOW", "Age_group"]

dx = df.set_index(ix)
deaths = dx.loc[:, "Deaths"].unstack("Age_group")
deaths = deaths.fillna(1)

pop = dx.loc[:, "Population"].unstack("Age_group")
1/0

pca = GPCA(deaths, 2, offset=offset, family=sm.families.Poisson())
r = pca.fit()