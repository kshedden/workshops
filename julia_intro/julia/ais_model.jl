# This script fits a basic regression model to the AIS data.
#
# See the comments in the `get_ais_data.jl` script or notebook for
# more information about the data.
#
# To run this script, enter the following in a julia session:
#
# ~~~
# include("ais_model.jl")
# ~~~
#
# If you get a `LoadError` when running this script, most likely you
# have not installed one or more packages that are used by the script.
# Follow the instructions on the screen to install the missing
# packages using `Pkg.add`.

using DataFrames, GLM

# Read the data.  This code is included in another script so that it
# can be re-used elsewhere.
include("read_ais_data.jl")

# Reduce to a dataset of vessels.  Take the maximum speed as a measure of how fast
# the vessel is capable of travelling.  Length, width, and draft should be constant
# within vessels, so taking the maximum should be the same as taking any value.
g = x -> count(ismissing, x) == length(x) ? missing : maximum(skipmissing(x))
maxspeed = by(df, :VesselName, :SOG=>g, :Length=>g, :Width=>g, :Draft=>g)

# Rename the variables to something shorter
rename!(maxspeed, :SOG_function=>:SOG, :Length_function=>:Length, :Width_function=>:Width,
        :Draft_function=>:Draft)

# Drop rows with missing data
maxspeed = maxspeed[completecases(maxspeed), :]

# Drop vessels with zero length, draft, width, or speed.  These are probably
# ships that did not configure their AIS device properly.
maxspeed = filter(x->x.SOG>0 && x.Length>0 && x.Width>0 && x.Draft>0, maxspeed)

# A basic additive model
m1 = lm(@formula(SOG ~ Length  + Width + Draft), maxspeed)

# One way to get the R^2
r2_1 = cor(predict(m1), m1.model.rr.y)^2

# Log/log regression
m2 = lm(@formula(log(SOG) ~ log(Length)  + log(Width) + log(Draft)), maxspeed)
r2_2 = cor(predict(m2), m2.model.rr.y)^2
