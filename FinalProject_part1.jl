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
df = CSV.read("Dataset.csv", DataFrame)
ndata = size(df)[1]
println("There are $ndata data points in the dataset.")
countries = unique(df[!, "country"])
countries = vcat(countries, "ALL")

f_all = Figure(resolution = (1200, 800))
ax_all = Axis(f_all[1, 1], xlabel="Strength [MPa]", ylabel="GWP [kgCO2e/kg]", 
title = "Strength vs GWP", titlesize  = 40, ylabelsize = 30, xlabelsize = 30, 
xticks = 10:10:100, yticks = 0:0.1:1.0, xticksize = 20, yticksize = 20)

scatter!(ax_all, df[!, "strength [MPa]"], df[!, "gwp_per_kg [kgCO2e/kg]"], color=df[!, "gwp_per_kg [kgCO2e/kg]"], markersize=15)
f_all
save("f_all.png", f_all)

"""
#### Select data for training/testing
"""
#select data with MX as country
c = "US"
if c == "ALL"
    x_total = Float32.(collect(df[!, "strength [MPa]"]))
    y_total = Float32.(collect(df[!, "gwp_per_kg [kgCO2e/kg]"]))
else
    x_total = Float32.(collect(df[df[!, "country"].==c, :][!, "strength [MPa]"]))
    y_total = Float32.(collect(df[df[!, "country"].==c, :][!, "gwp_per_kg [kgCO2e/kg]"]))
end

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

    f_opt = Figure(resolution = (1200, 800))
    ax_opt = Axis(f_opt[1, 1], xlabel ="Concrete strength [MPa]", ylabel="GWP [kgCO2e/kg]",
    xlabelsize = 30, ylabelsize = 30, titlesize = 40)
    scatter!(ax_opt, x_total, y_total, color=:gray) #plot all data
    scatter!(ax_opt, x_opt , y_opt, color=:blue,  markersize = 15) # plot lower bound
    scatter!(ax_opt, x_pes, y_pes, color=:red, marker = :square, markersize = 10) # plot upper bound
    f_opt
end
f_opt


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


#DO for India and Mexico

#India
#select data with IN as country
c = "IN"
x_total = Float32.(collect(df[df[!, "country"].==c, :][!, "strength [MPa]"]))
y_total = Float32.(collect(df[df[!, "country"].==c, :][!, "gwp_per_kg [kgCO2e/kg]"]))

data = hcat(x_total, y_total)
f_IN = Figure(resolution = (1200,800))
ax_IN = Axis(f_IN[1, 1], xlabel ="Concrete strength [MPa]", ylabel="GWP [kgCO2e/kg]", title = "India",
xlabelsize = 30, ylabelsize = 30, titlesize = 40)
xlims!(ax_IN,0,75)
ylims!(ax_IN,0,.4)
scatter!(ax_IN, x_total, y_total, color=:gray) #plot all data
f_IN

model_IN =  Chain(Dense(1, 50, sigmoid),Dense(50,50,sigmoid), Dense(50, 1))
model_IN, lh_IN, th_IN = train_model!(model_IN,data ,data, ϵ = 1e-6)
xval = range(minimum(x_total), stop=maximum(x_total), length=100)
xval_ = [[x] for x in xval]
yval = [ i[1] for i in model_IN.(xval_)]
lines!(ax_IN, xval, yval, color=:red)
f_IN
save("IN.png", f_IN)

#Mexico
#select data with MX as country
c = "MX"
x_total = Float32.(collect(df[df[!, "country"].==c, :][!, "strength [MPa]"]))
y_total = Float32.(collect(df[df[!, "country"].==c, :][!, "gwp_per_kg [kgCO2e/kg]"]))
x_total = sort(x_total)
y_total = sort(y_total)
deleteat!(x_total, 2)
deleteat!(y_total, 2)

data = hcat(x_total, y_total)
f_MX = Figure(resolution = (1200,800))
ax_MX = Axis(f_MX[1, 1], xlabel ="Concrete strength [MPa]", ylabel="GWP [kgCO2e/kg]", title = "Mexico",
xlabelsize = 30, ylabelsize = 30, titlesize = 40)
xlims!(ax_MX,0,90)
ylims!(ax_MX,0,.4)
scatter!(ax_MX, x_total, y_total, color=:gray) #plot all data
f_MX

model_MX =  Chain(Dense(1, 50, sigmoid),Dense(50,50,sigmoid), Dense(50, 1))
model_MX, lh_MX, th_MX = train_model!(model_MX,data ,data, ϵ = 1e-5)
xval = range(minimum(x_total), stop=maximum(x_total), length=100)
xval_ = [[x] for x in xval]
yval  = [ i[1] for i in model_MX.(xval_)]
lines!(ax_MX, xval, yval, color=:red)
f_MX
save("MX.png", f_MX)

function get_func(c::String, df::DataFrame)
   
    x_total = Float32.(collect(df[df[!, "country"].==c, :][!, "strength [MPa]"]))
    y_total = Float32.(collect(df[df[!, "country"].==c, :][!, "gwp_per_kg [kgCO2e/kg]"]))
    
    data = hcat(x_total, y_total)
    f_MX = Figure(resolution = (1200, 800))
    ax_MX = Axis(f_MX[1, 1], xlabel ="Concrete strength [MPa]", ylabel="GWP [kgCO2e/kg]", title = "Mexico",
    xlabelsize = 30, ylabelsize = 30, titlesize = 40)
    scatter!(ax_MX, x_total, y_total, color=:gray) #plot all data
    f_MX

    model_MX =  Chain(Dense(1, 50, sigmoid),Dense(50,50,sigmoid), Dense(50, 1))
    model_MX, lh_MX, th_MX = train_model!(model_MX,data ,data, ϵ = 1e-5)
    xval = range(minimum(x_total), stop=maximum(x_total), length=100)
    xval_ = [[x] for x in xval]
    yval = [ i[1] for i in model_MX.(xval_)]
    lines!(ax_MX, xval, yval, color=:red)
    f_MX

    return model_MX , f_MX
end

f2g_MX = x-> model_MX([x])[1]
f2g_IN = x-> model_IN([x])[1]

f2g_MX(20)
f2g_IN(20)

x = [ 10 ,20, 30]
y = f2g_MX.(x)

#try plot the functions together. 
f_func = Figure(resolution = (1200, 800))
f_func
ax_func = Axis(f_func[1, 1], xlabel ="Concrete strength [MPa]", ylabel="GWP [kgCO2e/kg]",
xlabelsize = 30, ylabelsize = 30, titlesize = 40,
xticks= 0:10:100, yticks = 0:0.1:0.7)
xlims!(ax_func, 5, 105)
ylims!(ax_func, 0, 0.7)

# x_total = df[!, "strength [MPa]"]
# y_total = df[!, "gwp_per_kg [kgCO2e/kg]"]

scatter!(ax_func, x_total, y_total, color=:gray, label= "data") #plot all data

lines!(ax_func, xval, y_pred_pes_sig, color=:blue , linestyle = :dash, label = " pes")
#plot MX
lines!(ax_func, xval, f2g_MX.(xval), color=:green , linestyle = :solid, label = " MX")
#plot in
lines!(ax_func, xval, f2g_IN.(xval), color=:orange , linestyle = :solid, label = " IN")
lines!(ax_func, xval, y_pred_opt_sig, color=:red , linestyle = :dash, label = " opt")


f_func[1,2]= Legend(f_func, ax_func, framevisible = false)

f_func

save("func.png", f_func)

println("PART 1 DONE")