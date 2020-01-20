# Download AIS data for analysis.
#
# AIS is a system for tracking ships:
#
#   https://en.m.wikipedia.org/wiki/Automatic_identification_system
#
# AIS data for ships travelling near the US coastline are available
# from the US government from this site:
#
#   https://marinecadastre.gov/ais/
#
# There is a separate data file for each year/month/zone, where zone
# is a geographical zone numbered from 1 to 20.  A map of the zones is
# here:
#
#   https://marinecadastre.gov/AIS/AIS%20Documents/UTMZoneMap2014.png
#
# The data files have URL's with the following format:
#
#   https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/AIS_2017_01_Zone01.zip
#
# To run this script, use the following line in a julia session:
#
#   include("get_ais_data.jl")
#
# You may get a LoadError if you are missing some packages.  Follow the
# commands on the screen for using Pkg.add to resolve this.
#
# Before using this script, you should set the data_dir variable below
# to point to a directory that you can write to.  Also, you can modify
# the year, month, and zone ranges below to customize how much data
# are obtained.

using Printf

# The download directory.  This must be writeable by you and have enough
# space to store the files.  You can delete the zip directory within
# data_dir after running this script.
data_dir = "/nfs/kshedden/AIS"

# This is the base url for all data files
base_url = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler"

# The subdirectory of data_dir/raw that will contain the data files
# after the script completes running.
ais_dir = "AIS_ASCII_by_UTM_Month"

# Create directories for storing the raw zip files and the extracted
# csv files.
mkpath("$data_dir/zip")
mkpath("$data_dir/raw")

# Download a collection of AIS data archive files in InfoZip format,
# extract the csv files from them, and compress them using gzip.
function julia_download()

    # Process this range of years
    for year in 2016:2016

        # Process this range of months
        for month in 1:1

            # Process this range of zones
            for zone in 10:10

                # Form the url as a string
                b = @sprintf("AIS_%4d_%02d_Zone%02d.zip", year, month, zone)
                u = @sprintf("%s/%4d/%s", base_url, year, b)

                # Download the data from the url.  The file is a zip file
                # (i.e. an InfoZip archive).
                cmd = `wget $u -O $data_dir/zip/$b`
                run(cmd)

                # Unzip the file.  Each zip archive should contain exactly
                # one csv file, which is extracted to the 'raw' directory
                # of data_dir.
                cmd = `unzip -o $data_dir/zip/$b -d $data_dir/raw`
                run(cmd)
            end
        end
    end

    # Gzip compress all the csv files
    for (root, dirs, files) in walkdir("$data_dir/raw")
        for f in files
            if splitext(f)[2] == ".csv"
                p = joinpath(root, f)
                cmd = `gzip -f $p`
                run(cmd)
            end
        end
    end

    # Some of the years are stored in a directory named YYYY_v2,
    # e.g. 2017_v2.  Create a symlink with only the year to make
    # it easier to access the files by year.
    for f in readdir("$data_dir/raw/$ais_dir")
        if occursin("_", f)
            v = split(f, "_")
            year = v[1]
            if !ispath("$data_dir/raw/$ais_dir/$year")
                symlink("$data_dir/raw/$ais_dir/$f", "$data_dir/raw/$ais_dir/$year")
            end
        end
    end

end

download()
