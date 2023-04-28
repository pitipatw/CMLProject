# Loading and setup of required packages
using KernelFunctions
using LinearAlgebra
using Distributions

# Plotting
using Plots;
default(; lw=2.0, legendfontsize=11.0, ylims=(-150, 500));

using Random: seed!
seed!(42);


f_truth(x) = (x + 4) * (x + 1) * (x - 1) * (x - 3)

x_train = -5:0.5:5
x_test = -7:0.1:7

noise = rand(Uniform(-20, 20), length(x_train))
y_train = f_truth.(x_train) + noise
y_test = f_truth.(x_test)

plot(x_test, y_test; label=raw"$f(x)$")
scatter!(x_train, y_train; seriescolor=1, label="observations")

function linear_regression(X, y, Xstar)
    weights = (X' * X) \ (X' * y)
    return Xstar * weights
end;

y_pred = linear_regression(x_train, y_train, x_test)
scatter(x_train, y_train; label="observations")
plot!(x_test, y_pred; label="linear fit")

function featurize_poly(x; degree=1)
    return repeat(x, 1, degree + 1) .^ (0:degree)'
end

function featurized_fit_and_plot(degree)
    X = featurize_poly(x_train; degree=degree)
    Xstar = featurize_poly(x_test; degree=degree)
    y_pred = linear_regression(X, y_train, Xstar)
    scatter(x_train, y_train; legend=false, title="fit of order $degree")
    return plot!(x_test, y_pred)
end

plot((featurized_fit_and_plot(degree) for degree in 1:4)...)

plot((featurized_fit_and_plot(degree) for degree in 1:4)...)
#what does "..." do at the end of the plot function?

