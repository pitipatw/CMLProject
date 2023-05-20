# for Topology optimization
# using TopOpt, LinearAlgebra, StatsFuns
# for Data Visualization
using Makie, GLMakie
# for Data Analysis
using CSV, DataFrames
using Clustering
#These are for the surrogate model
using Optimisers
using Flux, MLJ
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


"""
## Load data
"""
df = CSV.read("Dataset_1.csv", DataFrame)
ndata = size(df)[1]
println("There are $ndata data points in the dataset.")
countries = unique(df[!, "country"])
countries = vcat(countries, "ALL")

f_all = Figure(resolution = (1200, 800))
ax_all = Axis(f_all[1, 1], xlabel="Strength [MPa]", ylabel="GWP [kgCO2e/kg]")
ax_all.title = "Strength vs GWP"
ax_all.titlesize  = 40
ax_all.ylabelsize = 30
ax_all.xlabelsize = 30
scatter!(ax_all, df[!, "strength [MPa]"], df[!, "gwp_per_kg [kgCO2e/kg]"], color=:blue, markersize=20)
f_all
save("f_all.png", f_all)

"""
#### Select data for training/testing
"""
#select data with MX as country
c = "US"
x_total = Float32.(collect(df[df[!, "country"].==c, :][!, "strength [MPa]"]))
y_total = Float32.(collect(df[df[!, "country"].==c, :][!, "gwp_per_kg [kgCO2e/kg]"]))
#find the upper and lower bound
opt_pts = find_lowerbound(x_total, y_total)
pes_pts = find_upperbound(x_total, y_total)

#convert to matrix
opt = Matrix{Float32}(undef, length(opt_pts), 2)
pes = Matrix{Float32}(undef, length(pes_pts), 2)
for i in eachindex(opt_pts)
    opt[i, :] = [opt_pts[i][1], opt_pts[i][2]]
end

for i in eachindex(pes_pts)
    pes[i, :] = [pes_pts[i][1], pes_pts[i][2]]
end

if c == "US" #remove the first one, it's kinda buggy
    opt = opt[2:end, :]
    pes = pes[2:end, :]
end

#plot qmodel to check
begin
    x_opt = opt[:, 1]
    y_opt = opt[:, 2]
    x_pes = pes[:, 1]
    y_pes = pes[:, 2]

    f_opt = Figure(resolution = (1200, 700))
    ax_opt = Axis(f_opt[1, 1], xlabel ="Concrete strength [MPa]", ylabel="GWP [kgCO2e/kg]",
    xlabelsize = 30, ylabelsize = 30, titlesize = 40)
    scatter!(ax_opt, x_total, y_total, color=:gray) #plot all data
    scatter!(ax_opt, x_opt , y_opt, color=:blue,  markersize = 15) # plot lower bound
    scatter!(ax_opt, x_pes, y_pes, color=:red, marker = :square, markersize = 10) # plot upper bound
    f_opt
end



# x_max = maximum(data[:, 1])
# x_min = minimum(data[:, 1])
# y_max = maximum(data[:, 2])
# y_min = minimum(data[:, 2])

# @assert un_normalize_data(normalize_data(train_data, x_max,x_min, y_max,y_min),x_max,x_min, y_max,y_min ) == train_data

# train_data_n = normalize_data(train_data, x_max,x_min, y_max,y_min)
# test_data_n = normalize_data(test_data, x_max,x_min, y_max,y_min)


qmodel_opt_sig =  Chain(Dense(1, 50, sigmoid),Dense(50,50,sigmoid), Dense(50, 1))
qmodel_opt_relu = Chain(Dense(1, 50, relu),Dense(50,50,relu), Dense(50, 1))
qmodel_opt_tanh = Chain(Dense(1, 50, tanh),Dense(50,50,tanh), Dense(50, 1))

qmodel_opt_sig, loss_history1, test_history1 = train_model!(qmodel_opt_sig, opt, opt, ϵ = 1e-6)
qmodel_opt_relu, loss_history2, test_history2 = train_model!(qmodel_opt_relu, opt, opt, ϵ = 1e-6)
qmodel_opt_tanh, loss_history3, test_history3 = train_model!(qmodel_opt_tanh, opt, opt, ϵ = 1e-6)

predict_opt_sig = x -> qmodel_opt_sig([x])[1]
predict_opt_relu = x -> qmodel_opt_relu([x])[1]
predict_opt_tanh = x -> qmodel_opt_tanh([x])[1]

xval = range(minimum(x_total), stop=maximum(x_total), length=100)
y_pred_opt_sig =  [ i[1] for i in predict_opt_sig.(xval)]
y_pred_opt_relu =  [ i[1] for i in predict_opt_relu.(xval)]
y_pred_opt_tanh =  [ i[1] for i in predict_opt_tanh.(xval)]


lines!(ax_opt, xval, y_pred_opt_sig, color=:blue, label = "sigmoid")
lines!(ax_opt, xval, y_pred_opt_relu, color=:green, label = "relu")
lines!(ax_opt, xval, y_pred_opt_tanh, color=:orange, label = "tanh")
f_opt

#do the same with pes data

qmodel_pes_sig = Chain(Dense(1, 50, sigmoid),Dense(50,50,sigmoid), Dense(50, 1))
qmodel_pes_relu = Chain(Dense(1, 50, relu),Dense(50,50,relu), Dense(50, 1))
qmodel_pes_tanh = Chain(Dense(1, 50, tanh),Dense(50,50,tanh), Dense(50, 1))

qmodel_pes_sig, loss_history4, test_history4 = train_model!(qmodel_pes_sig, pes, pes, ϵ = 1e-6)
qmodel_pes_relu, loss_history5, test_history5 = train_model!(qmodel_pes_relu, pes, pes, ϵ = 1e-6)
qmodel_pes_tanh, loss_history6, test_history6 = train_model!(qmodel_pes_tanh, pes, pes, ϵ = 1e-6)

predict_pes_sig = x -> qmodel_pes_sig([x])[1]
predict_pes_relu = x -> qmodel_pes_relu([x])[1]
predict_pes_tanh = x -> qmodel_pes_tanh([x])[1]

y_pred_pes_sig =  [ i[1] for i in predict_pes_sig.(xval)]
y_pred_pes_relu =  [ i[1] for i in predict_pes_relu.(xval)]
y_pred_pes_tanh =  [ i[1] for i in predict_pes_tanh.(xval)]

lines!(ax_opt, xval, y_pred_pes_sig, color=:blue , linestyle = :dash, label = "sigmoid")
lines!(ax_opt, xval, y_pred_pes_relu, color=:green, linestyle = :dash, label = "relu")
lines!(ax_opt, xval, y_pred_pes_tanh, color=:orange, linestyle = :dash, label = "tanh")


legend = ["data", "opt sig", "opt relu", "opt tanh", "pes sig", "pes relu", "pes tanh"]
f_opt[1, 2] = Legend(f_opt, ax_opt, "Activation function", framevisible = false)
f_opt

save("activation_functions.png", f_opt, px_per_unit=:px, width=:auto, height=:auto)

###Let's end here.









#=================================================================================#
#let's get data that's more than 10% (val) of the opt and pes function 


#at add each point that's close to the opt and pes function than the specified distance.
#if the point is already in the list, don't add it again
#will do Thursday night.
#=================================================================================#

#### Separate the data into training and testing
data = hcat(x_total, y_total); # data is a 2 x n matrix
train_data, test_data = MLJ.partition(data, 0.7, multi=true, rng=100)# rng = Random.seed!(1234))

#construct models
f_near_opt , data_opt = get_nearest(data, predict_opt_tanh, 0.05)
f_near_opt

f_near_pes , data_pes = get_nearest(train_data, predict_pes_tanh, 0.05)
f_near_pes

# data_opt = hcat(x_opt, y_opt); # data is a 2 x n matrix
# data_pes = hcat(x_pes, y_pes); # data is a 2 x n matrix

train_data_opt, test_data_opt = MLJ.partition(data_opt, 0.7, multi=true, rng=100)# rng = Random.seed!(1234))
train_data_pes, test_data_pes = MLJ.partition(data_pes, 0.7, multi=true, rng=100)# rng = Random.seed!(1234))




println("#"^50)
println("There are $(size(train_data)[1]) data points in the training set.")
println("There are $(size(test_data)[1]) data points in the testing set.")
println("#"^50)

println("#"^50)
println("There are $(size(train_data_opt)[1]) data points in the training set.")
println("There are $(size(test_data_opt)[1]) data points in the testing set.")
println("#"^50)

println("#"^50)
println("There are $(size(train_data_pes)[1]) data points in the training set.")
println("There are $(size(test_data_pes)[1]) data points in the testing set.")
println("#"^50)





# x_max = maximum(data[:, 1])
# x_min = minimum(data[:, 1])
# y_max = maximum(data[:, 2])
# y_min = minimum(data[:, 2])

# # @assert un_normalize_data(normalize_data(train_data, x_max,x_min, y_max,y_min),x_max,x_min, y_max,y_min ) == train_data

# train_data_pes_n = normalize_data(train_data_pes, x_max,x_min, y_max,y_min)
# test_data_pes_n = normalize_data(test_data_pes, x_max,x_min, y_max,y_min)
# train_data_opt_n = normalize_data(train_data_opt, x_max,x_min, y_max,y_min)
# test_data_opt_n = normalize_data(test_data_opt, x_max,x_min, y_max,y_min)
# # 
# #plot data Distributions
# f_dis = Figure(resolution=(800, 600)) 
# ax_dis = Axis(f_dis[1, 1], xlabel="x", ylabel="y", title="Data Distribution")
# hist(x_total, bins=20, alpha=0.5, label="x", color=:red)
# hist(test_data_n[:, 1], bins=20, alpha=0.5, label="x_train", color=:blue)



#### Construct models
Random.seed!(1234567)

@show (models, m_names) = constructModels()
m_names = [ string(i) for i in m_names]

# save_model, save_loss,save_test_loss = train_all!(models, train_data, test_data, ϵ = 1e-6)
save_model_opt, save_loss_opt, save_test_loss_opt = train_all!(models, train_data_opt, test_data_opt, ϵ = 1e-6)
f_loss_opt, f_func_opt, save_func_g_opt = plot_loss_func(save_model_opt, save_loss_opt,  train_data, m_names, ftitle = "(Opt)")
f_loss_opt
f_func_opt




save_model_pes, save_loss_pes, save_test_loss_pes = train_all!(models, train_data_pes, test_data_pes, ϵ = 1e-6)
f_loss_pes, f_func_pes, save_func_g_pes = plot_loss_func(save_model_pes, save_loss_pes,  train_data, m_names, ftitle = "(Pes)")
f_loss_pes
f_func_pes


#plot the surrogate model
save("f_func_opt.png", f_func_opt)
save("f_loss_opt.png", f_loss_opt)
save("f_func_pes.png", f_func_pes)
save("f_loss_pes.png", f_loss_pes)


# This is for the opt
f2e = x -> sqrt.(x) #normalized modulus
f2g = save_func_g_opt[2]