using Flux, Zygote
using SurrogatesFlux, Surrogates
using Statistics
using Random
using Distributions

x_train =[ 1, 3, 5, 7, 10]
y_train = [58, 79, 95, 100, 122]

model = Chain(Dense(1, 20, sigmoid), Dense(20, 1))
loss(x, y) = Fßlux.mse(model(x), y)
# Training of the neural network
learning_rate = 0.01
optimizer = Descent(learning_rate)  # Simple gradient descent. See Flux documentation for other options.
n_epochs = 500
bounds = Float32[0.0], Float32[10.0]

sgt = NeuralSurrogate(x_train, y_train, bounds..., model=model, loss=loss, opt=optimizer, n_echos=n_epochs)
sgt.(x_test)[:]


# Testing the new model
x_test = Surrogates.sample(30, bounds..., SobolSample())[:]
x_test = [x[1] for x in x_test]
# test_error = mean(abs2, sgt(x) - f(x) for x in x_test)

#plot the original data and randomly sample data
f1 = Figure(resolution = (1200,800))
ax1 = Axis(f1[1,1], xlabel = "x1", ylabel = "x2")
x1 = [x[1] for x in x_train]
scatter!(ax1 , x_train, y_train , label="original data")
scatter!(ax1 , x_test, sgt.(x_test) , label="randomly sampled data")
f1


#using Flux built in
model = Chain(Dense(1, 50, leakyrelu), Dense(50, 1))
model[1].weight
model[2].weight
model[1].bias
model[2].bias

loss(model, x,y) = Flux.mse(model(x), y)
loss(model, x_train, y_train)


x_train =[ 1 3  5  7 10]
y_train = [58 79 95 100 122]
using Flux: train!
opt = Descent(0.0001)
data = [(x_train, y_train)]
for i = 1:1000
    train!(loss, Flux.params(model), data, opt);
    # model[1].weight
    println(loss(model, x_train', y_train'))
end
loss(model, x_train', y_train')
#plot the original data and randomly sample data
f2 = Figure(resolution = (1200,800))
ax2 = Axis(f2[1,1], xlabel = "x1", ylabel = "x2")
scatter!(ax2 , x_train, y_train , label="original data")
scatter!(ax2 , x_test, [x[1] for x in model.([[x] for x in x_test])], label="randomly sampled data")
f2

function f2g(x)
    return [4700*sqrt(x[1]) for x in model.([[x] for x in x])]
end

f2g(x_test)

function f2e(x)
    return [x[1] for x in model.([[x] for x in x])]
end

f2e(x_test)