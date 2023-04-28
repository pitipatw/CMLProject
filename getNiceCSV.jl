#tidy up files into dataframes
using DataFrames, CSV
using JSON

using Makie, GLMakie, GeoMakie

include("utilities.jl")
"""
Todo
Select only US or NewZealand, or somewhere that has dense data points.
Unpack those Vector{Vector{...}} into columns 
raw_data is a defualt -> find a way to save it as a variable file to load later.
using FileIO, JLD2
a = 1
FileIO.save("myfile.jld2","a",a)
b = FileIO.load("myfile.jld2","a")

"""

filepath = joinpath(@__DIR__, "Scraping/all_files/")

global total_data = Vector{Any}()
total_pages = 1384 #find a way to get number of total files in the folder
pages = 1:total_pages
# pages = 1:2 test
@time for i in pages
    page_num = string(i)
    filename = "ECS_page_" * page_num * ".json"
    vec1 = Vector{Any}() #reset
    open(filepath * filename, "r") do f
        txt = JSON.read(f, String)  # file information to string
        vec1 = JSON.parse(txt)  # parse and transform data

    end
    global total_data = vcat(total_data, vec1)
end

#should get 138400 x 328
@time df_core = DataFrame(total_data)
df = deepcopy(df_core)
# begin
#     #save the file
#     using FileIO, JLD2
#     @time FileIO.save("28042023df.jld2","df_core",df_core)
#     loaded_df = FileIO.load("28042023df.jld2","df_core")
#     @assert loaded_df == df_core

# end
#Will work on df, df_core is a backup.



#go through each column in df
#if the value in that column is a vector 
# create a new column, with the first column name follows by "_" then the sub column name 
# if the value is null, or false, turn them into either "" or 0.0, or 0, depend on the type of that column





df_for_csv = copy(df[!, list_of_names])
for i in names(df_for_csv)
    for j in df_for_csv[!, i]
        if j !== nothing
            println("____")
            println(i)
            println(j)
            println(typeof(j))
            break
        end
    end
end

CSV.write("EC3.csv", df_for_csv)




df_mod = DataFrame()

#loop gwp column, split it into 2 columns
#the first column will be parsed to Float64, the second column will be a string.
#the second column will be the unit of the first column
#add those to df_mod gwp and gwp_unit




splitNum_Unit(df, "gwp")

headers = names(df)


# ultimately, these can be put in a single (or two) vectors to loop
headers_with_string = [
    "standard_deviation",
    "gwp_per_category_declared_unit",
    "best_practice",
    "uncertainty_adjusted_gwp",
    "declared_unit",
    "mass_per_declared_unit",
    "conservative_estimate",
    "concrete_compressive_strength_28d",
    "lowest_plausible_gwp",
    "gwp_per_kg",
    "density",
    "gwp"
]


headers_in_category_with_string = [
    "pct10_gwp",
    "pct20_gwp",
    "pct30_gwp",
    "pct40_gwp",
    "pct50_gwp",
    "pct60_gwp",
    "pct70_gwp",
    "pct80_gwp",
    "pct90_gwp"
]

headers_in_plant_or_group_with_string = [ 
    "carbon_intensity"
]

headers_in_plant_or_group_string = [
    "name",
    "address",
    "country",
    "id",
]

headers_in_plant_of_group_float = [
    "latitude",
    "longitude"
]

headers_in_pog_owned_by_string= [
    "name",
]

headers_in_pog_owned_by_float = [
    "latitude",
    "longitude"
]

headers_in_manufacturer = [
    "name",
    "latitude",
    "longitude"
]


headers_general_string = [ 
        "warnings",
        "created_on",
        "description",

]
headers_float = [
"gwp_z",
"concrete_self_consolidating",

]
header_bool = [
    "fiber_reinforced",

]
for i in headers_with_string
    if i in headers #in case of typo
        n = i*"_numeric"
        u = i*"_unit"
        df_mod[!, n], df_mod[!, u] = splitNum_Unit(df, i)
    else
        println("column $i not found")
    end
end

#going into the category
category_keys = collect(keys(df.category[1]))
for i in headers_in_category_with_string
    if i in category_keys 
        n = i*"_numeric"
        u = i*"_unit"
        df_mod[!, n], df_mod[!, u] = splitNum_Unit(df, "category", i)
    else
        println("column $i not found")
    end
end

#latidude and longitude of the plants and the owners
plant_or_group_keys = collect(keys(df.plant_or_group[1]))
#this time they are pure float
for i in headers_in_plant_of_group_float
    V = Vector{Float64}(undef, length(df.plant_or_group))
    if i in plant_or_group_keys 
        n = i
        # println(i)
        for j in 1:length(df[!,1])
            # println(j)
            if i ∉ keys(df.plant_or_group[j])
                println("row $j, column $i not found")
                println("using owner's location")
                if df.plant_or_group[j]["owned_by"][i] === nothing
                    V[j] = 0.0
                else
                    V[j] = df.plant_or_group[j]["owned_by"][i]
                end
            else
                # println(df.plant_or_group[j][i])
                if df.plant_or_group[j][i] === nothing
                    V[j] = 0.0
                else
                    V[j]= df.plant_or_group[j][i]
                end
            end
        end
        df_mod[!, n] = V
    else
        println("column $i not found")
    end
end


for i in headers_in_plant_or_group_string
    V = Vector{String}(undef, length(df.plant_or_group))
    if i in plant_or_group_keys 
        n = i
        # println(i)
        for j in 1:length(df[!,1])
            # println(j)
            if i ∉ keys(df.plant_or_group[j])
                println("row $j, column $i not found")
                println("using owner's country")
                if df.plant_or_group[j]["owned_by"][i] === nothing
                    V[j] = "missing"
                else
                    V[j] = df.plant_or_group[j]["owned_by"][i]
                end
            else
                # println(df.plant_or_group[j][i])
                if df.plant_or_group[j][i] === nothing
                    V[j] = "missing"
                else
                    V[j]= df.plant_or_group[j][i]
                end
            end
        end
        df_mod[!, n] = V
    else
        println("column $i not found")
    end
end



#check
fig1 = Figure(resolution=(1200, 800))
ax1  = Axis(fig1[1, 1])
hist( df_mod[!,"gwp_numeric"], bins = 100)
fig1

#Data visualization
#initialize a list of data
latitudes = Vector{Float64}()
longitudes = Vector{Float64}()

#tidy up the data
for i in df.plant_or_group
    if "latitude" in keys(i)
        if i["latitude"] === nothing || i["longitude"] === nothing
            push!(latitudes, 0)
            push!(longitudes, 0)
            println("row $i was tidied up")
        else
            push!(latitudes, i["latitude"])
            push!(longitudes, i["longitude"])
        end
    else
        push!(latitudes, 0)
        push!(longitudes, 0)
    end
end

df_mod[!, "latitude"] = latitudes
df_mod[!, "longitude"] = longitudes



concrete_strength_at28 = Vector{Float64}(undef, length(df.concrete_compressive_strength_28d))

concrete_strength_raw = df.concrete_compressive_strength_28d
#find element that is nothing
for i in 1:length(concrete_strength_raw)
    if concrete_strength_raw[i] === nothing
        concrete_strength_raw[i] = "0 MPa!"
    elseif concrete_strength_raw[i] === ""
        concrete_strength_raw[i] = "0 MPa!"
    elseif concrete_strength_raw[i] === "0"
        concrete_strength_raw[i] = "0 MPa!"
    end
end
dummy = 0
concrete_strength_raw = split.(concrete_strength_raw)
strength_unit = Vector{String}(undef, length(concrete_strength_raw))
#get only the first column of the array
for i in eachindex(concrete_strength_raw)
    if length(concrete_strength_raw[i]) == 2
        concrete_strength_at28[i] = parse(Float64, (concrete_strength_raw[i][1]))
        strength_unit[i] = concrete_strength_raw[i][2]
    elseif length(concrete_strength_raw[i]) == 1
        dummy = concrete_strength_raw[i]
        dummy = split(dummy[1], "M")
        concrete_strength_at28[i] = parse(Float64, dummy[1])
    end
end

concrete_strength_at28

df_mod[!, "concrete_compressive_strength_28d"] = concrete_strength_at28


gwp = Vector{Float64}(undef, length(df.gwp_per_category_declared_unit))
gwp_raw = Vector{String}(undef, length(df.gwp_per_category_declared_unit))
declared_unit_raw = Vector{String}(undef, length(df.gwp_per_category_declared_unit))
mass_per_declared_unit_raw = Vector{String}(undef, length(df.gwp_per_category_declared_unit))
pct50_raw = Vector{String}(undef, length(df.gwp_per_category_declared_unit))


declared_unit_raw = df.declared_unit
units = [i[2] for i in split.(declared_unit_raw, " ")]
df_mod[!, "units"] = units
df_mod[!, "numeric_units"] = [parse(Float64, i[1]) for i in split.(declared_unit_raw, " ")]




df_mod[!, "id"] = 1:length(df.id)



for i in eachindex(df.gwp)
    if length(df.gwp[i]) == 1
        gwp_raw[i] = "0 kgCO2e"
    else
        gwp_raw[i] = df.gwp[i]
    end
end


gwp_raw = split.(gwp_raw)
gwp = [parse(Float64, i[1]) for i in gwp_raw]
gwp_units = [i[2] for i in gwp_raw]
#get only the first column of the array
df_mod[!, "gwp"] = gwp
df_mod[!, "gwp_units"] = gwp_units
df_mod[!, "gwp_per_category_declared_unit"] = df_mod.gwp ./ df_mod.numeric_units

for i in eachindex(df.density)
    if df.density[i] === nothing
        df.density[i] = "2400 kg/m3"
    elseif df.density[i] === ""
        df.density[i] = "2400 kg/m3"
    end
end

density_num = [parse(Float64, i[1]) for i in split.(df.density, " ")]
df_mod[!, "density"] = density_num

#copy to modify
filtered_df = deepcopy(df_mod)
US_df = deepcopy(df_mod)
US_df = filter(:latitude => >(24), US_df)
US_df = filter(:latitude => <(50), US_df)
US_df = filter(:longitude => >(-125), US_df)
US_df = filter(:longitude => <(-65), US_df)
US_df = filter(:concrete_compressive_strength_28d => >(0), US_df)


filtered_df = filter(:units => ==("m3"), filtered_df)
filtered_df = filter(:concrete_compressive_strength_28d => >(0), filtered_df)
filtered_df = filter(:concrete_compressive_strength_28d => <(200), filtered_df)
filtered_df = filter(:latitude => !=(0), filtered_df)
filtered_df = filter(:longitude => !=(0), filtered_df)
fig = Figure(resolution=(1200, 800))
ax1 = GeoAxis(
    fig[1, 1]; # any cell of the figure's layout
    dest="+proj=wintri", # the CRS in which you want to plot
    coastlines=true # plot coastlines from Natural Earth, as a reference.
)
ax2 = Axis(fig[1, 1], xlabel="concrete strength [MPa]", ylabel="GWP [kgCO2e/kg]")
ax3 = Axis(fig[1, 1], xlabel="concrete strength [psi]", ylabel="GWP [kgCO2e/kg]", title="the US")
scatter!(ax1, filtered_df.longitude, filtered_df.latitude; color=filtered_df.concrete_compressive_strength_28d, markersize=20000 * filtered_df.gwp ./ filtered_df.numeric_units ./ filtered_df.density / 1000)
fig
#scatter plot between gwp and concrete strength
ax = Axis(fig[1, 1], xlabel="GWP [kgCO2e/m3]", ylabel="concrete strength [MPa]")
col = [exp(cosd(l)) + 3(y / 90) for (l, y) in zip(filtered_df.longitude, filtered_df.latitude)]
scatter!(ax2, filtered_df.concrete_compressive_strength_28d, filtered_df.gwp ./ filtered_df.numeric_units ./ filtered_df.density; color=col, markersize=10)
scatter!(ax3, US_df.concrete_compressive_strength_28d, US_df.gwp ./ US_df.numeric_units ./ US_df.density; color="red", markersize=10)