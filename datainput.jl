using CSV , DataFrames
using Surrogates
using Flux
using Statistics
using SurrogatesFlux
using Makie, GLMakie

#load csv file into a dataframe called df
df = CSV.read("readytorun2.csv", DataFrame)

df[!,"country"]
df_IN = df[df[!,"country"] .== "IN",:]
df_US = df[df[!,"country"] .== "US",:]
df_CA = df[df[!,"country"] .== "CA",:]
df_AU = df[df[!,"country"] .== "AU",:]
df_NZ = df[df[!,"country"] .== "NZ",:]
df_SG = df[df[!,"country"] .== "SG",:]


#take the strength [Mpa] and gwp_per_kg [kgCO2e/kg] for the train and test dataset. 




#Sampling
x = collect(df_IN[!,"strength [MPa]"])
y = collect(df_IN[!,"gwp_per_kg [kgCO2e/kg]"])
#get x train rancomly from X

x_train = x[rand(1:length(x), 100)]
#get y train rancomly from Y
y_train = y[rand(1:length(y), 100)]

x_train = Float32.(x)
y_train = Float32.(y)



# f = x -> x[1]^2 + x[2]^2
# bounds = Float32[-1.0, -1.0], Float32[1.0, 1.0]
bounds = Float32[0.0] , Float32[60.0]
# Flux models are in single precision by default.
# Thus, single precision will also be used here for our training samples.

# x_train = sample(100, bounds..., SobolSample())
# y_train = f.(x_train)

# Perceptron with one hidden layer of 20 neurons.
model = Chain(Dense(1, 20, relu), Dense(20, 1))
loss(X, Y) = Flux.mse(model(X),Y)

# Training of the neural network
learning_rate = 0.1
optimizer = Descent(learning_rate)  # Simple gradient descent. See Flux documentation for other options.
n_epochs = 1000
sgt = NeuralSurrogate(x_train, y_train, bounds..., model=model, loss=loss, opt=optimizer, n_echos=n_epochs)

# Testing the new model
# x_test = sample(30, bounds..., SobolSample())
# test_error = mean(abs2, sgt(x) - f(x) for x in x_test)

#plot the original data and randomly sample data
x_rand = sample(10,0,50,SobolSample())
y_rand = sgt.(x_rand)

fig = Figure(resolution = (1200,800))
ax1 = Axis(fig, xlabel = "strength [MPa]", ylabel = "gwp_per_kg [kgCO2e/kg]")
# scatter(x,y,label="original data")
scatter!(x_train,y_train, label= "train data" , color = :blue)
scatter!(x_rand,y_rand,label= "randomly sampled test data", color = :red)
scatter!(x_train,sgt.(x_train), label = "another", color = :green)



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




lb = 0.0
ub = 50.0


#Creating surrogate
alpha = 2.0
n = 2
my_lobachevsky = LobachevskySurrogate(x_train,y_train,lb,ub,alpha=alpha,n=n)
#Approximating value at 5.0



value = my_lobachevsky(55.0)

#plot the original data and randomly sample data
x_rand = sample(10,lb,ub,SobolSample())
y_rand = my_lobachevsky.(x_rand)
fig = Figure(resolution = (1200,800))
ax1 = Axis(fig, xlabel = "strength [MPa]", ylabel = "gwp_per_kg [kgCO2e/kg]")
scatter(x,y,label="original data")
scatter!(x_rand,y_rand,label="randomly sampled data")


#Adding more data points
surrogate_optimize(f,SRBF(),lb,ub,my_lobachevsky,UniformSample())

#New approximation
value = my_lobachevsky(5.0)
