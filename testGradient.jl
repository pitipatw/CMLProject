using Surrogates, SurrogatesFlux
using Flux
using Zygote

x_train = [1 2 3 4]
y_train = [10 30 80 90]

bounds = Float32[0.0] , Float32[5]
model = Chain(Dense(1, 50, sigmoid), Dense(50, 1))
loss(x, y) = Flux.mse(model(x),y) 

# Training of the neural network
learning_rate = 0.01
optimizer = Descent(learning_rate) 
n_epochs = 10
my_model = NeuralSurrogate(x_train, y_train, bounds..., model=model, loss=loss, opt=optimizer, n_echos=n_epochs)

m
f = x -> sum(m(x))

Zygote.gradient( f, [1] ) # This works
Zygote.gradient( f, 1) # This does not work

# dummy TopOpt objective
obj = y -> begin
return sum(f.(y))
end
# PseudoDensities(y)

# initial decision variables as a vector
# y0 = zeros(ncells * (nmats - 1))
y0 = 40*ones(10)
# testing the objective function
@show obj(y0)
# testing the gradient
Zygote.gradient(obj, y0)


