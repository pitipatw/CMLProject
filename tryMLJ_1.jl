using MLJ
using Flux
using MLJMultivariateStatsInterface
RidgeRegressor = @load RidgeRegressor pkg=MultivariateStats
pipe = Standardizer() |> RidgeRegressor(lambda=10)

X, y = @load_boston

mach = machine(pipe, X, y) |> fit!
yhat = MLJ.predict(mach, X)
training_error = l1(yhat, y) |> mean

Flux.gradient( x-> MLJ.predict(mach, x) , 10)

using MLJ
LinearRegressors = @load LinearRegressor pkg=MLJLinearModels
model = LinearRegressors()
X = [1 2 3 4 5]
y = [10 30 80 90 100]
mach = machine(model, X, y)
fit!(mach)


mach = fit!(machine(LinearRegressor(), X, y))
predict(mach, X)
fitted_params(mach)


model = Dense(1=>1)
#train model
# model = Chain(Dense(1, 50, sigmoid), Dense(50, 1))
loss(x, y) = Flux.mse(model(x),y)
# Training of the neural network
learning_rate = 0.01
optimizer = Descent(learning_rate)
n_epochs = 10
my_model = NeuralSurrogate(X, y, model=model, loss=loss, opt=optimizer, n_echos=n_epochs)
using ProgressLogging
@withprogress