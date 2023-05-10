using CSV , DataFrames
using Surrogates
using Flux
using Statistics
using SurrogatesFlux
using Makie, CairoMakie
using Random
#load csv file into a dataframe called df
df = CSV.read("readytorun3.csv", DataFrame)

df[!,"country"]
df_IN = df[df[!,"country"] .== "IN",:]
df_US = df[df[!,"country"] .== "US",:]
df_CA = df[df[!,"country"] .== "CA",:]
df_AU = df[df[!,"country"] .== "AU",:]
df_NZ = df[df[!,"country"] .== "NZ",:]
df_SG = df[df[!,"country"] .== "SG",:]

using Clustering

X = Matrix{Float32}(undef, 3,length(df_US[!,"strength [MPa]"]))
X[1,:] = Flux.normalize(collect(df_US[!,"strength [MPa]"]))
X[2,:] = Flux.normalize(collect(df_US[!,"gwp_per_kg [kgCO2e/kg]"]))
X[3,:] = Flux.normalize(collect(df_US[!,"slope"]))
k = 6
XX = reshape(X[2:3,:], :, length(X[3,:]))
R = kmeans(Matrix(X),k)
@assert nclusters(R) == k
##we have to normalize the data, so the cluster is more resonable.
a = assignments(R) 
c = counts(R)
M = R.centers

f1 = Figure(resolution = (1200,800))
ax2 = Axis(f1[1,1], xlabel = "strength [MPa]", ylabel = "gwp_per_kg [kgCO2e/kg]")
scatter!(ax2, X[1,:], X[2,:], color = R.assignments)
f1


#should add this into a column in the dataframe 
df_US[!,"Group"] = R.assignments

for i in 1:k 
    ax1  = (
        
    )
end


using LazySets

#take the strength [Mpa] and gwp_per_kg [kgCO2e/kg] for the train and test dataset. 

function getCluster(df::DataFrame; T::String = "opt")
    df_out = deepcopy(df)
    if T == "opt"
        df_out = df_out[df_out[!,"Group"] .== "C",:]
        points = Vector{Vector{Float64}}()
        for i in eachindex(df_out[!,"strength [MPa]"])
            str_i = df_out[i,"strength [MPa]"]
            gwp_i = df_out[i,"gwp_per_kg [kgCO2e/kg]"]
            points = push!(points,[str_i,gwp_i])
        end
    elseif T == "pes"
        df_out = df_out[df_out[!,"Group"] .== "A",:]
        points = Vector{Vector{Float64}}()
        for i in eachindex(df_out[!,"strength [MPa]"])
            str_i = df_out[i,"strength [MPa]"]
            gwp_i = df_out[i,"gwp_per_kg [kgCO2e/kg]"]
            points = push!(points,[str_i,gwp_i])
        end
    else 
        println("please enter a valid type: opt or pes")
    end
    hull = convex_hull(points)
return reduce(vcat,transpose.(points)), reduce(vcat,transpose.(hull))
end

v_m,hull_m = getCluster(df_US; T = "pes")

# points = N -> [randn(2) for i in 1:N]
# # v = points(30)
# # hull = convex_hull(v)

# #turn v into matrix
# v_m = reduce(vcat,transpose.(v))
# hull_m = reduce(vcat,transpose.(hull))

# f1 = Figure(resolution = (1200,800))
ax1 = Axis(f1[1,2], xlabel = "strength [MPa]", ylabel = "gwp_per_kg [kgCO2e/kg]")
scatter!(v_m[:,1], v_m[:,2] , label = "points")
scatter!(ax1,hull_m[:,1], hull_m[:,2], label = "convex hull", color = :red)
f1

list_of_groups = Vector{Array{Float64,1}}()
for i = 1:k 
    push!(list_of_groups, X[:,a .== i])
end
#now we have our dataset.
#split the data into train and test set 
ndata = size(v_m)[1]
ntrain = Int(round(ndata*0.8))
ntest = ndata - ntrain
x_train = v_m[:,1][rand(ndata)]

#Sampling
x = collect(df_US[!,"strength [MPa]"])
y = collect(df_US[!,"gwp_per_kg [kgCO2e/kg]"])
#get x train rancomly from X

x_train = copy(x[1:1000])
y_train = copy(y[1:1000])

f0 = Figure(resolution = (1200,800))
ax0 = Axis(f0, xlabel = "strength [MPa]", ylabel = "gwp_per_kg [kgCO2e/kg]")
scatter!(x_train,y_train,label="original data")

# f = x -> x[1]^2 + x[2]^2
# bounds = Float32[-1.0, -1.0], Float32[1.0, 1.0]
bounds = Float32[0.0] , Float32[60.0]
# Flux models are in single precision by default.
# Thus, single precision will also be used here for our training samples.

# x_train = sample(100, bounds..., SobolSample())
# y_train = f.(x_train)

# Perceptron with one hidden layer of 20 neurons.
model = Chain(Dense(1, 50, sigmoid), Dense(50, 1))
loss(X, Y) = Flux.mse(model(X),Y) 

# Training of the neural network
learning_rate = 0.01
optimizer = Descent(learning_rate)  # Simple gradient descent. See Flux documentation for other options.
n_epochs = 100
sgt = NeuralSurrogate(x_train, y_train, bounds..., model=model, loss=loss, opt=optimizer, n_echos=n_epochs)

# Testing the new model
# x_test = sample(30, bounds..., SobolSample())
# test_error = mean(abs2, sgt(x) - f(x) for x in x_test)

#plot the original data and randomly sample data
x_test = sample(100,5,100,UniformSample())
y_test = sgt.(x_test)

fig = Figure(resolution = (1200,800))
ax1 = Axis3(fig[1,1], xlabel = "strength [MPa]", ylabel = "gwp_per_kg [kgCO2e/kg]", zlabel = "NONE")
# scatter(x,y,label="original data")
scatter!(x_train,y_train, label= "train data" , color = :blue)
scatter!(x_test,y_test,label= "randomly sampled test data", color = :red)
scatter!(fig[1,1], x_train , sgt.(x_train), label = "another", color = :green)



f = x -> x[1]^2 + x[2]^2
bounds = Float32[-1.0, -1.0], Float32[1.0, 1.0]
# Flux models are in single precision by default.
# Thus, single precision will also be used here for our training samples.

x_train = sample(100, bounds..., SobolSample())
y_train = f.(x_train)

# Perceptron with one hidden layer of 20 neurons.
model = Chain(Dense(2, 20, relu), Dense(20, 1))
loss(x, y) = Flux.mse(model(x), y)

# Training of the neural network
learning_rate = 0.1
optimizer = Descent(learning_rate)  # Simple gradient descent. See Flux documentation for other options.
n_epochs = 50
sgt = NeuralSurrogate(x_train, y_train, bounds..., model=model, loss=loss, opt=optimizer, n_echos=n_epochs)

# Testing the new model
x_test = sample(30, bounds..., SobolSample())
test_error = mean(abs2, sgt(x)[1] - f(x) for x in x_test)

#plot the original data and randomly sample data
f2 = Figure(resolution = (1200,800))
ax2 = Axis3(f2, xlabel = "x1", ylabel = "x2", zlabel = "f(x1,x2)")
x1 = [x[1] for x in x_train]
x2 = [x[2] for x in x_train]
scatter( x1, x2, y_train , label="original data")
x1 = [x[1] for x in sample(1000, bounds..., SobolSample())]
x2 = [x[2] for x in sample(1000, bounds..., SobolSample())]
x12  = [(x1[i],x2[i]) for i in 1:length(x1)]
scatter!(x1,x2, sgt.(x12), label="randomly sampled data")


include("mySurrogate.jl")

model
model([1])



