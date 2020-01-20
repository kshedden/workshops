# The utility script loads the AIS data into a dataframe.
#
# You should run the `get_ais_data.jl` script (or notebook) to
# download data before running this script.  Also, the `data_dir`
# variable in this script should be set to match the `data_dir`
# variable in `get_ais_data.jl`.  Finally, the `data_year` and
# `data_file` variables below should be set to the name of one of the
# data files that you want to analyze.
#
# If you get a `LoadError` when running this script, most likely you
# have not installed one or more packages that are used by the script.
# Follow the instructions on the screen to install the missing
# packages using `Pkg.add`.

# Package imports

using DataFrames, CSV, GZip

# This variable should be set to the same value as in the get_ais_data.jl
# script.

data_dir = "/nfs/kshedden/AIS"

# The year for the data file to be analyzed, as a string.

data_year = "2016"

# The name of the data file to be analyzed.

data_file = "AIS_2016_01_Zone10.csv.gz"

# A directory name baked into the zip archives, do not change.

ais_dir = "AIS_ASCII_by_UTM_Month"

# The full path to the data file

fn = joinpath(data_dir, "raw", ais_dir, data_year, data_file)

# Specify column types for certain key variables.  The symbol => in Julia
# is called the "Pair" function.  Writing x=>y and Pair(x, y) are equivalent.
# The pair function is used whenever it is natural to match two values in
# an ordered pair.

ty = Dict("LAT"=>Float64, "LON"=>Float64, "HEADING"=>Float64, "Length"=>Float64,
          "Width"=>Float64, "Draft"=>Float64, "SOG"=>Float64, "COG"=>Float64,
          "VesselType"=>Int64)

# Read the data file. The "do" pattern creates a block so that the file
# handle is closed immediately after the block is execuated.  The df
# variable name has to be defined as something (anything) outside the
# block, otherwise the df variable is local to the block and cannot
# be used after the block closes.

df = nothing
GZip.open(fn) do f
    global df
    df = CSV.read(f, types=ty, silencewarnings=true)
end
