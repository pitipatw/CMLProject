#tidy up files into dataframes
using DataFrames
using JSON

filepath = joinpath(@__DIR__,"all_files/")

# treat a json object of arrays or array of objects as a "table"
#get a dummyinput file
jtable = jsontable(json_source)
#join that to the rest of the files.
#loop all files from page number.
# turn json table into DataFrame
df = DataFrame(jtable)