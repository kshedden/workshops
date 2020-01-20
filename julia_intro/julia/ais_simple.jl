# This script performs some basic analyses of the AIS data.
#
# See the comments in the `get_ais_data.jl` script or notebook for
# more information about the data.
#
# To run this script, enter the following in a julia session:
#
# ~~~
# include("ais_simple.jl")
# ~~~
#
# See these links for information about some of the data values:
#
# * https://www.navcen.uscg.gov/?pageName=AISMessagesA
# * https://www.navcen.uscg.gov/pdf/AIS/AISGuide.pdf
#
# If you get a `LoadError` when running this script, most likely you
# have not installed one or more packages that are used by the script.
# Follow the instructions on the screen to install the missing
# packages using `Pkg.add`.

# Package imports

using DataFrames, CSV, Clustering, StatsBase, GZip, Printf

# Read the data.  This code is included in another script so that it
# can be re-used elsewhere.

include("read_ais_data.jl")

# Count the number of unique vessels.  DataFrame columns can be accessed
# with the dot syntax used here for brevity.

n_vessels = size(unique(df.VesselName))

# This is equivalent to above, using a different syntax to access the
# dataframe column.  A token of the form :x is a called a "symbol".
# If you want to know more about symbols in julia, see here:
# https://stackoverflow.com/a/23482257/1941745

n_vessels = size(unique(df[:, :VesselName]))

# Look at the distribution of records over the status codes.  countmap
# is in StatsBase, which we imported above.

cm = countmap(df.Status)

# Get the maximum speed per vessel.  We write a custom aggregation
# function to handle missing values, and the special case where all
# values in a group are missing.

function g1(x)
    if count(ismissing, x) == length(x)
        return missing
    else
        return maximum(skipmissing(x))
    end
end
maxspeed1 = by(df, :VesselName, :SOG=>g1)

# This is equivalent to the preceeding cell, expressed more concisely.
# The syntax "x -> expression" creates an anonymous function that evaluates
# the expression using a provided value of x.

g2 = x -> count(ismissing, x) == length(x) ? missing : maximum(skipmissing(x))
maxspeed2 = by(df, :VesselName, :SOG=>g2)

# Confirm that the results are the same

nx = sum(skipmissing(maxspeed1[:, 2] .!= maxspeed2[:, 2]))

# Create a dataframe containing records for moored cargo vessels.

dx = filter(x -> !ismissing(x.Status) && x.Status == "moored", df)
dx = filter(x -> !ismissing(x.VesselType) && x.VesselType in [70, 71, 73, 74, 76], dx)

# We will do a cluster analysis to find where the moored cargo ships are located.
# This clustering is based only on the spatial position of each vessel, so we
# restrict to those variables here.

dd = dx[:, [:LAT, :LON]]

# A few of the records have missing position information, which the clustering
# algorithm can't easily handle.  So we drop those records here.

dd = dd[completecases(dd), :]

# The clustering code wants the positions in the form of a matrix, not a dataframe,
# so we do the conversion here.

dd = Matrix{Float64}(dd)

# The clustering algorithm wants the objects in the columns and the variables
# (coordinates) in the rows, so we transpose.

dd = transpose(dd)

# Use k-means clustering to locate some clusters of moored cargo ships

cl = kmeans(dd, 5)

# Get the sizes of the clusters and figure out how to sort the clusters
# by descending size.

ct = counts(cl)
ii = sortperm(ct, rev=true)

# Create urls that can be used to locate the clusters on a map.

pos = cl.centers
for i in ii
    @printf("https://www.google.com/maps/search/?api=1&query=%f,%f\n", pos[1, i], pos[2, i])
end
