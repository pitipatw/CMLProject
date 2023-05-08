using Flux

model = Chain(Dense(1 => 50, sigmoid), Dense(50 => 1))
model(5)
model([10])
Flux.gradient( model, Float32(10) ) # This works

obj = y -> begin
# _rhos = interp2(MultiMaterialVariables(y, nmats)) #rho is the density
ρ = f2g.(y)
# return sum(_rhos.x) / ncells # elements have unit volumes, 0.4 is the target.
return sum(ρ)/ncells
end
# PseudoDensities(y)

# initial decision variables as a vector
# y0 = zeros(ncells * (nmats - 1))
y0 = 40*ones(ncells)
# testing the objective function
@show obj(y0)
# testing the gradient
Zygote.gradient(obj, y0)

m = Chain(x -> x^2, x -> x+1)
m = Chain(Dense(1,22))
m([5]) # => 26


model2 = Chain(
  Dense(10 => 5, σ),
  Dense(5 => 2),
  softmax)

model2(rand(10)) 