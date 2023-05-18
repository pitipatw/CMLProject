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
c = "MX"
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


#plot qmodel to check
begin
    x_opt = opt[:, 1]
    y_opt = opt[:, 2]
    x_pes = pes[:, 1]
    y_pes = pes[:, 2]

    f_opt = Figure(resolution = (800, 600))
    ax_opt = Axis(f_opt[1, 1])
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

qmodel_opt_sig, loss_history, test_history = train_model!(qmodel_opt_sig, opt, opt, ϵ = 1e-6)
qmodel_opt_relu, loss_history, test_history = train_model!(qmodel_opt_relu, opt, opt, ϵ = 1e-6)
qmodel_opt_tanh, loss_history, test_history = train_model!(qmodel_opt_tanh, opt, opt, ϵ = 1e-6)

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

qmodel_pes_sig, loss_history, test_history = train_model!(qmodel_pes_sig, pes, pes, ϵ = 1e-6)
qmodel_pes_relu, loss_history, test_history = train_model!(qmodel_pes_relu, pes, pes, ϵ = 1e-6)
qmodel_pes_tanh, loss_history, test_history = train_model!(qmodel_pes_tanh, pes, pes, ϵ = 1e-6)

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
#=================================================================================#
#let's get data that's more than 10% (val) of the opt and pes function 

range_fc′ = 10:0.1:100
distance = 10 
#at add each point that's close to the opt and pes function than the specified distance.
#if the point is already in the list, don't add it again

#will do Thursday night.
#=================================================================================#

#### Separate the data into training and testing
data = hcat(x_total, y_total); # data is a 2 x n matrix
train_data, test_data = MLJ.partition(data, 0.7, multi=true, rng=100)# rng = Random.seed!(1234))

println("#"^50)
println("There are $(size(train_data)[1]) data points in the training set.")
println("There are $(size(test_data)[1]) data points in the testing set.")
println("#"^50)

#construct models

# x_max = maximum(data[:, 1])
# x_min = minimum(data[:, 1])
# y_max = maximum(data[:, 2])
# y_min = minimum(data[:, 2])

# @assert un_normalize_data(normalize_data(train_data, x_max,x_min, y_max,y_min),x_max,x_min, y_max,y_min ) == train_data

# train_data_n = normalize_data(train_data, x_max,x_min, y_max,y_min)
# test_data_n = normalize_data(test_data, x_max,x_min, y_max,y_min)

# #plot data Distributions
# f_dis = Figure(resolution=(800, 600)) 
# ax_dis = Axis(f_dis[1, 1], xlabel="x", ylabel="y", title="Data Distribution")
# hist(x_total, bins=20, alpha=0.5, label="x", color=:red)
# hist(test_data_n[:, 1], bins=20, alpha=0.5, label="x_train", color=:blue)



#### Construct models
Random.seed!(12346)
Random.seed!(1234567)

@show (models, m_names) = constructModels()

models[1][1].weight


save_model, save_loss,save_test_loss = train_all!(models, train_data, test_data, ϵ = 1e-6)


# for i in eachindex(models)
#     mn = i
#     selected_model = models[mn]
#     selected_model_name = m_names[mn]
#     println("Selected model: $selected_model_name")

#     selected_model, model_loss_history, test_loss_history = train_model!(selected_model, train_data, test_data, ϵ = 1e-6)

#     save_model[i] = deepcopy(selected_model)
#     save_loss[i]  = deepcopy(model_loss_history)
#     save_test_loss[i] = deepcopy(test_loss_history)
#     println("DONE TRAINING for $selected_model_name")
# end


#Plotting

m_names = [ string(i) for i in m_names]

save_func_e = Vector{Function}(undef, length(models))
save_func_g = Vector{Function}(undef, length(models))


f_func = Figure(resolution=(1200, 800))
ax_func = Axis(f_func[1, 1], xlabel="Concrete strength [MPa]", ylabel="GWP [kgCO2e/kg]", title="Prediction")
ax_func.xlabelsize = 30
ax_func.ylabelsize = 30
ax_func.titlesize  = 40
f_func

xmax = maximum(data[:,1])
xmin = minimum(data[:,1])
ymax = maximum(data[:,2])
ymin = minimum(data[:,2])
if size(data)[1] < 10
    ax_func.xticks = 0:1:xmax
    ax_func.yticks = 0:0.01:ymax
else
    ax_func.xticks = 0:10:xmax
    ax_func.yticks = 0:0.05:ymax
end

xval = collect(xmin:0.1:xmax)
xval_ = [ [x] for x in xval]

scatter!(ax_func , data[:,1], data[:,2], markersize=20, color=:green, label = "Original Data")

f_func



f_loss = plot_loss(save_model, save_loss, m_names)
f_loss


#plot the surrogate model


f_func[1 ,2] = Legend(f_func, ax_func, "Model", framevisible = false)


f_func


save("f_func.png", f_func)
save("f_loss.png", f_loss)



# f_with_sur = plot_country(df[df[!, "country"].==c, :], c, selected_model)


#pick sigmoid with 2 layers
#this should give te surrogate model used for TopOpt.2