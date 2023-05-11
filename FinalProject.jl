# for Topology optimization
using TopOpt, LinearAlgebra, StatsFuns
# for Data Visualization
using Makie, GLMakie
# for Data Analysis
using CSV, DataFrames
using Clustering
#These are for the surrogate model
using Optimisers
using Flux, Zygote, MLJ
using SurrogatesFlux, Surrogates
using Statistics
using Random
using Distributions
#style
using ProgressLogging
"""
inputs 
country name (string, abbreviation)
model number (N1, N2_1 , N2_2, N3_1, N3_2, N3_3)
"""

include("utilities.jl")
include("findbound.jl")
include("train_functions.jl")
## settings
Makie.inline!(true) # so Makie plots are in Jupyter notebook




#### Construct models
models = constructModels()
selected_model = models[4]
#### Construct loss functions
loss1(model, x, y) = Flux.mse(model(x), y)

"""
## Load data
"""
df = CSV.read("Dataset_1.csv", DataFrame)
ndata = size(df)[1]
println("There are $ndata data points in the dataset.")
countries = unique(df[!, "country"])
countries = vcat(countries, "ALL")
for i in countries 
    if i == "ALL"
        df_i = df
        f = plot_country(df_i, String(i))
    else
	    df_i = df[df[!, "country"].==i, :]
	    f = plot_country(df_i, String(i))
    end
end



"""
#### Select data for training/testing
"""
#select data with MX as country
c = "IN"
x_total = collect(df[df[!, "country"].==c, :][!, "strength [MPa]"]);
y_total = collect(df[df[!, "country"].==c, :][!, "gwp_per_kg [kgCO2e/kg]"]);

#find the upper and lower bound
opt_pts = find_lowerbound(x_total, y_total)
pes_pts = find_upperbound(x_total, y_total)

#convert to matrix
opt = Matrix{Float64}(undef, length(opt_pts), 2)
pes = Matrix{Float64}(undef, length(pes_pts), 2)
for i in eachindex(opt_pts)
    opt[i, :] = [opt_pts[i][1], opt_pts[i][2]]
end

for i in eachindex(pes_pts)
    pes[i, :] = [pes_pts[i][1], pes_pts[i][2]]
end




#plot qmodel to check
begin
    x_opt = opt[:, 1]
    y_opt = opt[:, 2]
    x_pes = pes[:, 1]
    y_pes = pes[:, 2]

    f_opt = Figure(resolution = (800, 600))
    ax_opt = Axis(f_opt[1, 1])
    scatter!(ax_opt, x_total, y_total, color=:gray) #plot all data
    scatter!(ax_opt, x_opt , y_opt, color=:blue) # plot lower bound
    scatter!(ax_opt, x_pes, y_pes, color=:red, marker = :square) # plot upper bound
    f_opt
end


# x_max = maximum(data[:, 1])
# x_min = minimum(data[:, 1])
# y_max = maximum(data[:, 2])
# y_min = minimum(data[:, 2])

# @assert un_normalize_data(normalize_data(train_data, x_max,x_min, y_max,y_min),x_max,x_min, y_max,y_min ) == train_data

# train_data_n = normalize_data(train_data, x_max,x_min, y_max,y_min)
# test_data_n = normalize_data(test_data, x_max,x_min, y_max,y_min)


qmodel = Chain(Dense(1, 50, sigmoid),Dense(50,50,sigmoid), Dense(50, 1))
selected_model, loss_history, test_history = train_model!(qmodel, opt, opt)

predict = x -> selected_model([x])[1]

y_pred =  [ i[1] for i in predict.(x_opt)]

lines!(ax_opt, x_opt, y_pred, color=:red)
f_opt


#=================================================================================#


#### Separate the data into training and testing
data = hcat(x_total, y_total); # data is a 2 x n matrix
train_data, test_data = MLJ.partition(data, 0.7, multi=true, rng=100)# rng = Random.seed!(1234))

println("#"^50)
println("There are $(size(train_data)[1]) data points in the training set.")
println("There are $(size(test_data)[1]) data points in the testing set.")
println("#"^50)

#construct models

#normalize dataset
function normalize_data(data, x_max, x_min, y_max, y_min)
    data[:, 1] = (data[:, 1] .- x_min) ./ (x_max - x_min)
    data[:, 2] = (data[:, 2] .- y_min) ./ (y_max - y_min)
    return data
end

function un_normalize_data(data, x_max, x_min, y_max, y_min)
    data[:, 1] = data[:, 1] .* (x_max - x_min) .+ x_min
    data[:, 2] = data[:, 2] .* (y_max - y_min) .+ y_min
    return data
end


x_max = maximum(data[:, 1])
x_min = minimum(data[:, 1])
y_max = maximum(data[:, 2])
y_min = minimum(data[:, 2])

@assert un_normalize_data(normalize_data(train_data, x_max,x_min, y_max,y_min),x_max,x_min, y_max,y_min ) == train_data


train_data_n = normalize_data(train_data, x_max,x_min, y_max,y_min)
test_data_n = normalize_data(test_data, x_max,x_min, y_max,y_min)
#plot data Distributions
f_dis = Figure(resolution=(800, 600)) 
ax_dis = Axis(f_dis[1, 1], xlabel="x", ylabel="y", title="Data Distribution")
hist(x_total, bins=20, alpha=0.5, label="x", color=:red)
hist(test_data_n[:, 1], bins=20, alpha=0.5, label="x_train", color=:blue)

selected_model = models[4]
selected_model, model_loss_history, test_loss_history = train_model!(selected_model, train_data, test_data)

# design variables are fc′
# assign model into function
f2e = x -> sqrt.(x) #normalized modulus
f2g = x -> un_normalize_data(selected_model([x]),x_max,x_min, y_max,y_min)#will have to broadcast later.
f2g = x -> selected_model([x])[1] #will have to broadcast later.

begin
#plot loss and test 
f_loss = Figure(resolution=(1200, 800))
ax1 = Axis(f_loss[1, 1], xlabel="Epoch", ylabel="Loss", yscale=log10, title = "Loss vs Epoch")
lin = lines!(ax1, model_loss_history, markersize=7.5, color=:red)
sca = scatter!(ax1, test_loss_history, markersize=7.5)
ax1.subtitle = "Loss is $(valid_loss)"
Legend(f_loss[1, 2],
    [sca, lin],
    ["testing loss_history", "training loss_history"])
ax1.xlabelsize = 30
ax1.ylabelsize = 30
ax1.titlesize = 40
ax1.yticklabelsize = 23
f_loss
end

begin
f_pva = Figure(resolution=(1200, 800))
ax_pva = Axis(f_pva[1, 1], xlabel="Predicted", ylabel="Acctual ")
ax_pva.title = "Actual vs Predicted GWP [kgCO2e/kg]"

scatter!(ax_pva, test_data[:, 2], [x[1] for x in f2g.(test_data[:, 1])], color=:red, markersize=10)
ln = lines!(ax_pva, test_data[:, 2], test_data[:, 2])

Legend(f_pva[1, 2],
    [ln],
    ["y=x"])
f_pva
end


#plot line compare with the actual data
f_with_sur = plot_country(df[df[!, "country"].==c, :], c, selected_model)