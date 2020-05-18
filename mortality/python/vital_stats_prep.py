# This script prepares CDC mortality data for statistical analysis.
#
# Running the script (with the run function) produces an analysis file
# containing the number of deaths (due to all causes), and the US
# population in sex x age x year x month strata.
#
# Each row of the resulting data set contains the number of deaths in
# a specific demographic stratum, in a
# particular month.  For example, a row may contain the number of
# deaths for 0-5 year old girls, in November of 2013.
#
# The mortality data come from here:
#     https://www.cdc.gov/nchs/nvss/mortality_public_use_data.htm
#
# The population data come from:
#     https://www2.census.gov
#
# To run this script, edit the 'target_dir' path below to point to a
# valid location on your system.

import shutil
import urllib.request as request
from contextlib import closing
import os
import gzip
import zipfile
import pandas as pd
import numpy as np

# All data are stored in this location
target_dir = "/nfs/kshedden/cdc_mortality"

# The directory for the raw mortality data
mort_raw_dir = os.path.join(target_dir, "mort_raw")

# The directory for the raw population data
pop_raw_dir = os.path.join(target_dir, "pop_raw")

# The data for the final processed data
final_dir = os.path.join(target_dir, "final")

# Create the directories if they do not exist
for p in (mort_raw_dir, pop_raw_dir, final_dir):
    if not os.path.exists(p):
        os.makedirs(p)

# The url pattern for the mortality data
mort_url = "ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/mortality/mortYYYYus.zip"

# The url pattern for the population data
pop_url0 = "https://www2.census.gov/programs-surveys/demo/tables/age-and-sex/YYYY"
pop_url = os.path.join(pop_url0, "age-sex-composition/YYYYgender_table1.csv")

# Get data for these years
firstyear = 2007
lastyear = 2018


def download_mortality():
    """
    Download the mortality data.
    """
    for year in range(firstyear, lastyear + 1):
        p = mort_url.replace("YYYY", str(year))
        dst = os.path.join(mort_raw_dir, "mort%4dus.zip" % year)
        with closing(request.urlopen(p)) as r:
            with open(dst, 'wb') as w:
                shutil.copyfileobj(r, w)


def decompress_mortality():
    """
    Unzip the mortality data.
    """
    for year in range(firstyear, lastyear + 1):
        arx = os.path.join(mort_raw_dir, "mort%4dus.zip" % year)
        with zipfile.ZipFile(arx) as zf:
            fn = zf.filelist[0].filename
            r = zf.open(fn)
            dst = os.path.join(mort_raw_dir, "%4d.txt.gz" % year)
            with gzip.open(dst, "w") as w:
                shutil.copyfileobj(r, w)


def download_population():
    """
    Download the population data.
    """
    for year in range(firstyear, lastyear + 1):

        p = pop_url.replace("YYYY", str(year))

        if year == 2002:
            p = os.path.join(pop_url0, "ppl-167/table1.csv")
            p = p.replace("YYYY", str(year))
        elif year == 2003:
            p = p.replace(".csv", ".1.csv")
        elif year < 2007:
            p = p.replace(".csv", "-1.csv")

        dst = os.path.join(pop_raw_dir, "%4d_pop.csv" % year)
        with closing(request.urlopen(p)) as r:
            with open(dst, 'wb') as w:
                shutil.copyfileobj(r, w)


def aggregate():
    """
    Create aggregated death totals per demographic cell x month x year.
    """

    # Residence, month, sex, age units, age value, day of week, year
    cs = [(19, 20), (64, 66), (68, 69), (69, 70), (70, 73), (101, 105)]

    # Aggregate by age within bins defined by these ages.  The bins are closed on
    # the left and open on the right, e.g. the first bin is [0, 5).  The bins
    # are set to match the population count data from the census bureau.
    age_cuts = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 100]

    dz = []

    for year in range(firstyear, lastyear + 1):

        fn = os.path.join(mort_raw_dir, "%4d.txt.gz" % year)
        df = pd.read_fwf(fn, colspecs=cs, header=None)
        df.columns = ["Res", "Month", "Sex", "Age_units", "Age_value", "Year"]

        # Age can be coded in different units, convert everything to years.
        df["Age"] = np.nan
        df.loc[df.Age_units == 1, "Age"] = df.Age_value
        df.loc[df.Age_units == 2, "Age"] = df.Age_value / 12
        df.loc[df.Age_units == 3, "Age"] = df.Age_value / 365.25
        df.loc[df.Age_units == 4, "Age"] = df.Age_value / (24 * 365.25)
        df.loc[df.Age_units == 5, "Age"] = df.Age_value / (60 * 24 * 365.25)
        df.loc[df.Age_value==999, "Age"] = np.nan

        # Exclude people who are not US residents, as they are not included in
        # the population data.
        df.loc[df.Res != 4, :]

        df["Age_grp"] = pd.cut(df.Age, age_cuts, right=False)

        da = df.groupby(["Year", "Month", "Sex", "Age_grp"]).size()
        da.name = "Deaths"
        da = da.reset_index()

        dz.append(da)

    dz = pd.concat(dz, axis=0)
    dz["Age_group"] = [str(x) for x in dz.Age_grp]

    # Clean up the age bin label
    def f(x):
        x = x.replace("[", "").replace(")", "")
        x = x.split(",")
        x = [y.strip() for y in x]
        x = [float(y) for y in x]
        x[1] -= 1
        return "%02d_%02d" % tuple(x)

    dz["Age_group"] = [f(x) for x in dz.Age_group]
    dz = dz.drop("Age_grp", axis=1)

    dz.to_csv(os.path.join(final_dir, "aggregated_mort.csv.gz"), index=None)


def prep_population():
    """
    Prepare the population data.  Ensure that it is mergeable 1-1 with the
    mortality data.
    """

    da = []
    for year in range(firstyear, lastyear + 1):
        fn = os.path.join(pop_raw_dir, "%4d_pop.csv" % year)
        df = pd.read_csv(fn, skiprows=7, header=None)
        df = df.iloc[0:18, :]
        df.columns = ["Age_group", "Both", "x", "Male", "y", "Female", "z1", "z2"]
        for x in ["Female", "Male"]:
            df.loc[:, x] = [float(y.replace(",", "")) for y in df.loc[:, x]]
        df = df.loc[:, ["Age_group", "Female", "Male"]]
        df.loc[:, "Year"] = year

        da.append(df)

    da = pd.concat(da, axis=0)

    def f(x):
        x = x.replace(".", "").replace(" to ", "_").replace(" years", "")
        x = x.strip()
        if x == "Under 5":
            x = "00_04"
        if x == "85 and over":
            x = "85_99"

        x = x.split("_")
        x = [int(y) for y in x]
        x = "%02d_%02d" % tuple(x)

        return x

    da["Age_group"] = [f(x) for x in da.Age_group]
    for x in "Female", "Male":
        da.loc[:, x] = da.loc[:, x].astype(np.int)
    da = da.loc[:, ["Year", "Age_group", "Female", "Male"]]

    fn = os.path.join(final_dir, "pop.csv")
    da.to_csv(fn, index=None)


def final_merge():
    """
    Merge the mortality and population data.
    """

    dp = pd.read_csv(os.path.join(final_dir, "pop.csv"))
    dm = pd.read_csv(os.path.join(final_dir, "aggregated_mort.csv.gz"))
    dp = dp.melt(id_vars=["Year", "Age_group"])
    dp = dp.rename(columns={"value": "Population", "variable": "Sex"})
    dm["Sex"] = dm["Sex"].replace({"F": "Female", "M": "Male"})

    mv = ["Year", "Age_group", "Sex"]
    dx = pd.merge(dm, dp, left_on=mv, right_on=mv, how="left")
    dx.loc[:, "Population"] *= 1000
    dx = dx.loc[dx.Year >= firstyear, :]

    for x in "Year", "Month", "Population":
        dx[x] = dx[x].astype(np.int)

    dx.to_csv(os.path.join(final_dir, "pop_mort.csv"), index=None)


def run():
    download_mortality()
    decompress_mortality()
    download_population()
    aggregate()
    prep_population()
    final_merge()
