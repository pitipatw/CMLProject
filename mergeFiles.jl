#tidy up files into dataframes
using DataFrames, JSONTables, CSV
using JSON
using Makie
 

filepath = joinpath(@__DIR__,"Scraping\\all_files\\")

filepath = "\\Scraping\\all_files\\"
filename = "ECS_page_1.json"
# treat a json object of arrays or array of objects as a "table"
#get a dummyinput file
#
open(filepath*"ECS_page_2.json" , "r") do f
    global dict3
    dicttxt = JSON.read(f,String)  # file information to string
    dict3=JSON.parse(dicttxt)  # parse and transform data
end

# using JSON3
# using DataFrames 
# ls = filter(endswith("json"),readdir("."))
# df = JSON3.read.(read.(ls,String)) |> DataFrame


global total_data = Vector{Any}()
total_pages = 1384
pages = 1:total_pages
for i in pages
    page_num = string(i)
    filename = "ECS_page_" * page_num * ".json"
    vec1 = Vector{Any}()
    open(filepath*filename, "r") do f
        txt = JSON.read(f,String)  # file information to string
        vec1=JSON.parse(txt)  # parse and transform data
    end
    total_data = vcat(total_data,vec1)
end

#should get 138400 x 328
df = DataFrame(total_data)

#go through each column in df
#if the value in that column is a vector 
# create a new column, with the first column name follows by "_" then the sub column name 
# if the value is null, or false, turn them into either "" or 0.0, or 0, depend on the type of that column
removed_columns = Vector{String}()
for i in names(df)
    remove_chk = false
    # println(i)
    #check if there is a vector in the column
    #criterias to remove a column

    if typeof(df[!,i]) == Vector{Vector{Any}}
        println("Column $i is a vector of vector{Any}")
        remove_chk = true
    elseif (df[!,i] .!= nothing) == 0
        println("Column $i is nothing")
        remove_chk = true

    elseif typeof(df[!,i]) == Vector{Any}
        println("Column $i is a vector of vector{Any}")
        remove_chk = true

    elseif typeof(df[!,i]) == Vector{Nothing}
        println("Column $i is a vector of vector{Any}")
        remove_chk = true

    
    elseif typeof(df[!,i]) == Vector{Union{Nothing,Bool}}
        if (df[!,i] .!= nothing) == 0
            println("Column $i is a vector of vector{Any}")
            remove_chk = true

        elseif sum(df[!,i].==true) == 0
            println("Column $i is a vector of vector{Any}")
            remove_chk = true

        else
            println("***Column $i is not removed from Bool/Nothing check")
            remove_chk = true

        end
    
    elseif typeof(df[!,i]) == Vector{Union{Nothing,Int64}}
        if (df[!,i] .!= nothing) == 0
            println("Column $i is a vector of vector{Any}")
            remove_chk = true

        elseif sum(df[!,i].==true) == 0
            println("Column $i is a vector of vector{Any}")
            remove_chk = true

        else
            println("***Column $i is not removed from Int64/Nothing check")
        end

    elseif typeof(df[!,i]) == Vector{Union{Nothing,Float64}}
        if (df[!,i] .!= nothing) == 0
            println("Column $i is a vector of vector{Any}")
            remove_chk = true

        elseif sum(df[!,i].==true) == 0
            println("Column $i is a vector of vector{Any}")
            remove_chk = true
        else
            println("***Column $i is not removed from Float64/Nothing check")
        end
    elseif typeof(df[!, i]) == Vector{Union{Nothing, String}}
        if (df[!,i] .!= nothing) == 0
            println("Column $i is a vector of string but has all nothing values")
            remove_chk = true

        elseif sum(df[!,i].!= "") == 0
            println("Column $i is a vector of string with all empty string values")
            remove_chk = true
        else
            println("***Column $i is not removed from String/Nothing check")
            #create new column with the name starts with i, and follow by _ and the sub column name
            # for j in names
        end
    elseif typeof(df[!,i]) == Vector{Union{Nothing, Vector{Any}}}
        if (df[!,i] .!= nothing) != 0
            println("Column $i is a vector of vector{Any} but has all nothing values")
            remove_chk = true
        end
    end

#could add more remove criteria here. 

    if remove_chk
        select!(df, Not(i))
        println("Column $i is removed")
        push!(removed_columns,i)
    end
end



for i in names(df)
    if df[!,i] isa Vector
        for j in df[!,i]
            if j isa Vector
                println("Column $i is a vector of vector{Any}")
            end
            
        end
    end
end

#create a new empty dataframe df_mod 


limit = 100
small_columns = []
for i in names(df) 
    if typeof(df[!,i]) != Float64 || typeof(df[!,i]) != Int64
        println(i)
        println(typeof(df[!,i]))
        # println(sum(df[!,i].!= nothing))
        if sum(df[!,i].!= nothing) <=limit
            if i ∉ small_columns 
                push!(small_columns,i)
            end
        end
    end
end

dummy = size(small_columns)[1]
println("Number of small columns (less than $limit): $dummy")

list_of_names = names(df)
sort!(list_of_names)

#go through every single elements of df, 
# check the type of each vector, to replace nothing with "" or 0.0 or 0
function replace_nothing!(df)
end


for i in names(df)
    dummy = df[!,i][1]
    if dummy isa Dict
    elseif dummy isa Vector
        for j in df[!,i]
            for k in j
                if typeof(k) == Float64
                    df[!,i] = replace(df[!,i], nothing => 0.0)
                elseif typeof(k) == Int64
                    df[!,i] = replace(df[!,i], nothing => 0)
                elseif typeof(k) == String
                    df[!,i] = replace(df[!,i], nothing => "None")
                elseif typeof(k) == Bool
                    df[!,i] = replace(df[!,i], nothing => false)
                end
            end
        end
    elseif dummy == nothing
    else
        println(i)
        for j in eachindex(dummy)
            if typeof(dummy[j]) == Float64
                df[!,i][j] = replace(df[!,i][j], nothing => 0.0)
            elseif typeof(dummy[j]) == Int64
                df[!,i] = replace(df[!,i][j], nothing => 0)
            elseif typeof(dummy[j]) == String
                df[!,i] = replace(df[!,i][j], nothing => "None")
            elseif typeof(dummy[j]) == Bool
                df[!,i] = replace(df[!,i][j], nothing => false)
            end
        end
    end
end



df



function Makie.plot(P::Type{<: AbstractPlot}, fig::Makie.FigurePosition, arg::Solution; axis = NamedTuple(), kwargs...)

    menu = Menu(fig, options = ["viridis", "heat", "blues"])

    funcs = [sqrt, x->x^2, sin, cos]

    menu2 = Menu(fig, options = zip(["Square Root", "Square", "Sine", "Cosine"], funcs))

    fig[1, 1] = vgrid!(
        Label(fig, "Colormap", width = nothing),
        menu,
        Label(fig, "Function", width = nothing),
        menu2;
        tellheight = false, width = 200)

    ax = Axis(fig[1, 2]; axis...)

    func = Node{Any}(funcs[1])

    ys = @lift($func.(arg.data))

    scat = plot!(ax, P, Attributes(color = ys), ys)

    cb = Colorbar(fig[1, 3], scat)

    on(menu.selection) do s
        scat.colormap = s
    end

    on(menu2.selection) do s
        func[] = s
        autolimits!(ax)
    end

    menu2.is_open = true

    return Makie.AxisPlot(ax, scat)
end

# nrow = size(df)[1]
# df_mod = DataFrame()
# for i in names(df)

#     if df[!,i][1] isa Dict
#         #initialize an array nrow x length of the dict
#         A = Array{Any}(undef,nrow,length(df[!,i][1]))
#         key_list = []
#         for k in keys(df[!,i][1])
#             println("====")
#             println(i)
#             push!(key_list,k)
#             println(k)
#         end
#     end
# end

#         for j in eachindex(df[!,i])
#             println(j)
#             #j is a dict
#             for (k,v) in df[!,i][j]
#                 #k is a sub name
#                 println(k)
#                 new_name = i * "_" * k
#                 println(v)
#                 if new_name ∉ names(df_mod)
#                     T = typeof(v)
#                     dummy = Vector{T}

#                     df_mod[!,new_name] =Vector{T,undef,size(df)[1]}
#                 end
#                 df_mod[!,new_name] = push!(df_mod[!,new_name],v)
#             end
#         end
#     end
# end

            
#         end
#     elseif df[!,i][1] isa Vector
#     else
#         df_mod[!,i] = copy(df[!,i])
#     end
# end

# for i in names(df)
#     println(i,"::", typeof(df[!,i]))
# end



f = Figure();
lines(f[1, 1], Solution(0:0.3:10))
scatter(f[1, 2], Solution(0:0.3:10))
f |> display





select!(df, Not(:dp_rating)) 



a = select!(df, Not([1]))
a = select!(df, Not([2]))
dfm = replace!(df,  => nothing)


CSV.write("EC3.csv", df)


#initialize a list of data
latitudes = Vector{Float64}()
longitudes = Vector{Float64}()
concrete_strength = Vector{Float64}()
gwp = Vector{Float64}()
declared_unit = Vector{String}()

for i in df.plant_or_group
    push!(latitudes, i["latitude"])
    push!(longitudes, i["longitude"])
end
concrete_strength_raw = df.concrete_compressive_strength_28d
#find element that is nothing
for i in 1:length(concrete_strength)
    if concrete_strength_raw[i] == nothing
        concrete_strength_raw[i] = "0"
    end
end
concrete_strength = split.(concrete_strength_raw)
#get only the first column of the array
concrete_strength = [ parse(Float64, (i[1])) for i in concrete_strength]

gwp_raw = df.gwp_per_category_declared_unit

gwp = split.(gwp_raw)
#get only the first column of the array
gwp = [ parse(Float64, (i[1])) for i in gwp]
declared_unit_raw = df.declared_unit
declared_unit = split.(declared_unit_raw)
#get only the first column of the array
declared_unit = [ parse(Int64, i[1]) for i in declared_unit]
norm_gwp = gwp ./ declared_unit
# latitudes = df.plant_or_group[1]["latitude"]


##
using GeoMakie , GLMakie
lons = -180:180
lats = -90:90
slons = rand(lons, 2000)
slats = rand(lats, 2000)
sfield = [exp(cosd(l)) + 3(y/90) for (l,y) in zip(slons, slats)]

fig = Figure()
ax = GeoAxis(fig[1,1])
scatter!(longitudes, latitudes; color = concrete_strength, markersize = norm_gwp./maximum(norm_gwp)*10)
fig
##

#scatter plot between gwp and concrete strength
ax = Axis(fig[1,1], xlabel = "GWP [kgCO2e/m3]", ylabel = "concrete strength [MPa]")
scatter!(ax, gwp, concrete_strength, markersize = norm_gwp./maximum(norm_gwp)*10)

#do latitude, longitude, concrete strength at 28 days , gwp gwp_per_category_declared_unit, declared declared_unit

"cementitious"
"recycled_content" 
compressive_strength
fire_rating
standard_deviation
created_on
updated_on

compressive_strength
concrete_aggregate_size_max
concrete_air_entrain  #worth keeping
concrete_co2_entrain #worth keeping
concrete_compressive_strength_28d #good
concrete_compressive_strength_other
concrete_compressive_strength_other_d
concrete_flexion_strength
concrete_max_slump
concrete_min_pipeline_size
concrete_min_slump
concrete_self_consolidating
concrete_slump
concrete_w_c_ratio
gwp #good
gwp_per_category_declared_unit #good
gwp_per_kg
gwp_z
mass_per_declared_unit
name 
owner
recycled_content
declared_unit #good
zzz.plant_or_group[1]["address_line"]
zzz.plant_or_group[1]["latitude"]
zzz.plant_or_group[1]["longitude"]

lowest_plausible_gwp
data = JSON.parse(filepath*filename)
jtable = jsontable(filepath*filename)
#join that to the rest of the files.
#loop all files from page number.
# turn json table into DataFrame
df = DataFrame(jtable)