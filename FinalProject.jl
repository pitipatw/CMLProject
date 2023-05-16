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

# for i in countries 
#     if i == "ALL"
#         df_i = df
#         f = plot_country(df_i, String(i))
#     else
# 	    df_i = df[df[!, "country"].==i, :]
# 	    f = plot_country(df_i, String(i))
#     end
# end



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


qmodel_opt_sig = Chain(Dense(1, 50, sigmoid),Dense(50,50,sigmoid), Dense(50, 1))
qmodel_opt_relu = Chain(Dense(1, 50, relu),Dense(50,50,relu), Dense(50, 1))
qmodel_opt_tanh = Chain(Dense(1, 50, tanh),Dense(50,50,tanh), Dense(50, 1))

qmodel_opt_sig, loss_history, test_history = train_model!(qmodel_opt_sig, opt, opt, ϵ = 1e-9)
qmodel_opt_relu, loss_history, test_history = train_model!(qmodel_opt_relu, opt, opt, ϵ = 1e-9)
qmodel_opt_tanh, loss_history, test_history = train_model!(qmodel_opt_tanh, opt, opt, ϵ = 1e-9)

predict_opt_sig = x -> qmodel_opt_sig([x])[1]
predict_opt_relu = x -> qmodel_opt_relu([x])[1]
predict_opt_tanh = x -> qmodel_opt_tanh([x])[1]

y_pred_opt_sig =  [ i[1] for i in predict_opt_sig.(x_opt)]
y_pred_opt_relu =  [ i[1] for i in predict_opt_relu.(x_opt)]
y_pred_opt_tanh =  [ i[1] for i in predict_opt_tanh.(x_opt)]


lines!(ax_opt, x_opt, y_pred_opt_sig, color=:blue, label = "sigmoid")
lines!(ax_opt, x_opt, y_pred_opt_relu, color=:green, label = "relu")
lines!(ax_opt, x_opt, y_pred_opt_tanh, color=:orange, label = "tanh")
f_opt

#do the same with pes data

qmodel_pes_sig = Chain(Dense(1, 50, sigmoid),Dense(50,50,sigmoid), Dense(50, 1))
qmodel_pes_relu = Chain(Dense(1, 50, relu),Dense(50,50,relu), Dense(50, 1))
qmodel_pes_tanh = Chain(Dense(1, 50, tanh),Dense(50,50,tanh), Dense(50, 1))

qmodel_pes_sig, loss_history, test_history = train_model!(qmodel_pes_sig, pes, pes, ϵ = 1e-9)
qmodel_pes_relu, loss_history, test_history = train_model!(qmodel_pes_relu, pes, pes, ϵ = 1e-9)
qmodel_pes_tanh, loss_history, test_history = train_model!(qmodel_pes_tanh, pes, pes, ϵ = 1e-9)

predict_pes_sig = x -> qmodel_pes_sig([x])[1]
predict_pes_relu = x -> qmodel_pes_relu([x])[1]
predict_pes_tanh = x -> qmodel_pes_tanh([x])[1]

y_pred_pes_sig =  [ i[1] for i in predict_pes_sig.(x_pes)]
y_pred_pes_relu =  [ i[1] for i in predict_pes_relu.(x_pes)]
y_pred_pes_tanh =  [ i[1] for i in predict_pes_tanh.(x_pes)]

lines!(ax_opt, x_pes, y_pred_pes_sig, color=:blue , linestyle = :dash, label = "sigmoid")
lines!(ax_opt, x_pes, y_pred_pes_relu, color=:green, linestyle = :dash, label = "relu")
lines!(ax_opt, x_pes, y_pred_pes_tanh, color=:orange, linestyle = :dash, label = "tanh")


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
Random.seed!(12345)

@show (models, m_names) = constructModels()

models[1][1].weight

trained_model = Vector{Chain}(undef, length(models))

save_loss = Vector{Vector{Float32}}(undef, length(models))
save_test_loss = Vector{Vector{Float32}}(undef, length(models))

for i in eachindex(models)
    mn = i
    selected_model = models[mn]
    selected_model_name = m_names[mn]
    println("Selected model: $selected_model_name")

    selected_model, model_loss_history, test_loss_history = train_model!(selected_model, train_data, test_data, ϵ = 1e-6)

    save_model[i] = deepcopy(selected_model)
    save_loss[i]  = deepcopy(model_loss_history)
    save_test_loss[i] = deepcopy(test_loss_history)
    println("DONE TRAINING for $selected_model_name")
end


#Plotting

m_names = [ string(i) for i in m_names]

save_func_e = Vector{Function}(undef, length(models))
save_func_g = Vector{Function}(undef, length(models))

f_func = Figure(resolution=(1200, 800))
ax_func = Axis(f_func[1, 1], xlabel="x", ylabel="y", title="Prediction")
ax_func.xlabelsize = 30
ax_func.ylabelsize = 30
ax_func.titlesize  = 40
f_func

xmax = maximum(data[:,1])
ymax = maximum(data[:,2])
if size(data)[1] < 10
    ax_func.xticks = 0:1:xmax
    ax_func.yticks = 0:0.01:ymax
else
    ax_func.xticks = 0:10:xmax
    ax_func.yticks = 0:0.05:ymax
end

xval = collect(10:0.1:xmax)
xval_ = [ [x] for x in xval]


f_loss = Figure(resolution=(1200, 800))
ax_loss = Axis(f_loss[1, 1], xlabel="Epoch", ylabel="Loss", yscale=log10, xscale = log10, title = "Loss vs Epoch")
ax_loss.xlabelsize = 30
ax_loss.ylabelsize = 30
ax_loss.titlesize  = 40
#loop and plot all the models
for i in eachindex(save_model)
    model = save_model[i]
    name = m_names[i]

    # design variables are fc′
    # assign model into function
    f2e = x -> sqrt.(x) #normalized modulus
    f2g = x -> model([x])[1] #will have to broadcast later.
    save_func_e[i] = deepcopy(f2e)
    save_func_g[i] = deepcopy(f2g)
    

    #get line type
    # line_type = :solid
    # println(string(name[1]))
    w = 3
    if string(name[1]) == "1"
        col = :black
        line_type = :solid
        continue
    elseif string(name[1]) == "2" 
        col = :red
        if string(name[end]) == "d"
            line_type = :solid
        elseif string(name[end]) == "u"
            line_type = :dot
        elseif string(name[end]) == "h"
            line_type = :dash
            col = :green
        end

    elseif string(name[1]) == "3"
        col = :blue
        if string(name[end]) == "d"
            line_type = :solid
        elseif string(name[end]) == "u"
            line_type = :dot
        elseif string(name[end]) == "h"
            line_type = :dash
        end
    end

	lines!(ax_func, xval, [x[1] for x in model.(xval_)], color=col, linestyle= line_type, linewidth= w, label = name)
    # lines!(ax_func, range_fc′, f2g(range_fc'), markersize=7.5, color=col, linestyle = line_type, label = name)
    lines!(ax_loss, save_loss[i], markersize=7.5, color=col, linestyle = line_type, label = name, linewidth = 5)
    lines!(ax_loss, save_test_loss[i], markersize=7.5, color=col, linestyle = line_type, label = "test_"*name, linewidth = 2)

end


scatter!(ax_func , df_c[!, "strength [MPa]"], df_c[!, "gwp_per_kg [kgCO2e/kg]"], markersize=20, color=:green, label = "Original Data")

f_func
f_loss

f_func[1 ,2] = Legend(f_func, ax_func, "Model", framevisible = false)
f_loss[1, 2] = Legend(f_loss, ax_loss, "Model", framevisible = false)
f_loss

f_func


save("f_func.png", f_func)
save("f_loss.png", f_loss)



f_with_sur = plot_country(df[df[!, "country"].==c, :], c, selected_model)


#pick sigmoid with 2 layers
#this should give te surrogate model used for TopOpt.2