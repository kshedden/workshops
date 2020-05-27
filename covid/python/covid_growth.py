# https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

today = datetime.today().strftime('%Y-%m-%d')
today = pd.to_datetime(today)

df = pd.read_csv("ecdpc.csv")

def f(x):
    u = x.split("/")
    return "%s-%s-%s" % tuple(u[::-1])
df["date_rep"] = df.dateRep.apply(f)
df["date"] = pd.to_datetime(df.date_rep)

df = df.sort_values(["countriesAndTerritories", "date"])
df = df.loc[df.date <= today]

df["logDeaths"] = np.log(1 + df.deaths)
df["logCases"] = np.log(1 + df.cases)

df["cumDeaths"] = df.groupby("countriesAndTerritories")["deaths"].transform(np.cumsum)

df["days"] = (df.date - pd.to_datetime("2019-12-01")).dt.days

df = df.loc[df.cumDeaths >= 10, :]

df["rdays"] = df.groupby("countriesAndTerritories")["days"].transform(lambda x: x - x.min())

def slope(z):

    n = z.shape[0]
    if n < 20:
        return pd.Series(np.nan * np.zeros(n, dtype=np.float64))

    bw = 10
    dslope = np.zeros(n, dtype=np.float64)
    cslope = np.zeros(n, dtype=np.float64)

    for i in range(n):
        i1 = max(i-bw, 0)
        i2 = min(i+bw, n)
        x = z.rdays.iloc[i1:i2]
        y = z.logDeaths.iloc[i1:i2]
        c = np.cov(y, x)
        dslope[i] = c[0, 1] / c[1, 1]
        y = z.logCases.iloc[i1:i2]
        c = np.cov(y, x)
        cslope[i] = c[0, 1] / c[1, 1]

    return pd.DataFrame({"rdays": z.rdays, "dslope": dslope, "cslope": cslope})

dx = df.groupby("countriesAndTerritories").apply(slope)
dx = dx.reset_index()
dx = dx.drop(["level_1", 0], axis=1)
v = ["countriesAndTerritories", "rdays"]
df = pd.merge(df, dx, left_on=v, right_on=v, how="left")

plt.clf()
plt.grid(True)
for _, g in df.groupby("countriesAndTerritories"):
    plt.plot(g.rdays, g.dslope, '-', color='grey', alpha=0.4)

plt.clf()
plt.grid(True)
for _, g in df.groupby("countriesAndTerritories"):
    plt.plot(g.rdays, g.cslope, '-', color='grey', alpha=0.4)
