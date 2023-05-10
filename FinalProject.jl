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

include("utilities.jl")
## settings
Makie.inline!(true) # so Makie plots are in Jupyter notebook


## Load data
df = CSV.read("Dataset_1.csv", DataFrame)
ndata = size(df)[1]
println("There are $ndata data points in the dataset.")

countries = unique(df[!, "country"])
countries = vcat(countries, "ALL")
for i in countries 
    if i == "ALL"
        df_i = df
        f = plot_country(df_i, String(i))
    else
	    df_i = df[df[!, "country"].==i, :]
	    f = plot_country(df_i, String(i))
    end
end




#### Select data for training/testing
#select data with MX as country
c = "MX"
x_total = collect(df[df[!, "country"].==c, :][!, "strength [MPa]"]);
y_total = collect(df[df[!, "country"].==c, :][!, "gwp_per_kg [kgCO2e/kg]"]);

#### Separate the data into training and testing
data = hcat(x_total, y_total); # data is a 2 x n matrix
train_data, test_data = MLJ.partition(data, 0.7, multi=true, rng=100)# rng = Random.seed!(1234))
test_data, valid_data = MLJ.partition(test_data, 0.5, rng=123)
println("#"^50)
println("There are $(size(train_data)[1]) data points in the training set.")
println("There are $(size(test_data)[1]) data points in the testing set.")
println("There are $(size(valid_data)[1]) data points in the validation set.")
println("#"^50)

#### Construct models
N1 = Dense(1, 1)
N2_1 = Chain(Dense(1, 10, sigmoid), Dense(10, 1))
N2_2 = Chain(Dense(1, 10, relu), Dense(10, 1))
N3_1 = Chain(Dense(1, 10,), Dense(10, 10, sigmoid), Dense(10, 1))
N3_2 = Chain(Dense(1, 10,), Dense(10, 10, relu), Dense(10, 1))

global models = [N1, N2_1, N2_2, N3_1, N3_2]

#### Construct loss functions
loss1(model, x, y) = Flux.mse(model(x), y)

# loss2(N2,x,y) = Flux.mse(N2(x),y)
# loss3(x,y) = Flux.mse(N3(x),y)
# loss4(x,y) = Flux.mse(N4(x),y)

# loss_history = [loss1, loss2, loss3, loss4]



epoch = 500
loss_history = Float32[]
test_history = Float32[]
x_train = train_data[:, [1]]'
y_train = train_data[:, [2]]'

rule = Optimisers.Adam()  # use the Adam optimiser with its default settings
state_tree = Optimisers.setup(rule, N1);  # initialise this optimiser's momentum etc.
begin
    for i = 1:1000
        global N1, state_tree
        dLdm, _, _ = gradient(loss1, N1, x_train, y_train)
        state_tree, N1 = Optimisers.update(state_tree, N1, dLdm)
        val = loss1(N1, x_train, y_train)
        testval = loss1(N1, test_data[:, [1]]', test_data[:, [2]]')
        # println(val)
        push!(loss_history, val)
        push!(test_history, testval)
    end
    valid_loss=  loss1(N1, valid_data[:, [1]]', valid_data[:, [2]]')
    println("The validation loss is $valid_loss")
end
# design variables are fc′
# assign model into function
f2e = x -> sqrt.(x) #normalized modulus
f2g = x -> N1([x])


#plot loss and test 
f1 = Figure(resolution=(1200, 800))
ax1 = Axis(f1[1, 1], xlabel="Epoch", ylabel="Loss", yscale=log10, title = "Loss vs Epoch")
lin = lines!(ax1, loss_history, markersize=7.5, color=:red)
sca = scatter!(ax1, test_history, markersize=7.5)
ax1.subtitle = "Loss is $(valid_loss)"
Legend(f1[1, 2],
    [sca, lin],
    ["testing loss_history", "training loss_history"])
ax1.xlabelsize = 30
ax1.ylabelsize = 30
ax1.titlesize = 40
ax1.yticklabelsize = 23
f1


f_pva = Figure(resolution=(1200, 800))
ax_pva = Axis(f_pva[1, 1], xlabel="Predicted", ylabel="Acctual ")
ax_pva.title = "Actual vs Predicted GWP [kgCO2e/kg]"

scatter!(ax_pva, valid_data[:, 2], [x[1] for x in f2g.(valid_data[:, 1])], color=:red, markersize=10)
ln = lines!(ax_pva, valid_data[:, 2], valid_data[:, 2])

Legend(f_pva[1, 2],
    [ln],
    ["y=x"])
f_pva


#plot line compare with the actual data
f5 = Figure(resolution=(1200, 800))
ax3 = Axis(f5[1, 2], xlabel="Strength [MPa]", ylabel="GWP [kgCO2e/kg]")
ax3.title = "Strength vs GWP predicted plot"
ax3.titlesize = 40
xval = 12:1:80
lines!(ax_MX, xval, [x[1] for x in f2g.(xval)], color=:red, markersize=3)
scatter!(ax5, df_MX[!, "strength [MPa]"], df_MX[!, "gwp_per_kg [kgCO2e/kg]"], color=:red, markersize=10)
f5
f3
f_MX



f_IN
save("INwithSur.png", f_IN)



#### setup Topology Optimization for continuum #####
save("MXwithSur.png", f_MX)
#get benchmark problem
include("Benchmark1.jl")


compliance_threshold = 1000 # maximum compliance


E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 2.0 # downward force
rmin = 2.0 # filter radius
xmin = 0.0001 # minimum density
problem_size = (60, 20)
x0 = vcat(fill(1.0, prod(problem_size)), fill(100.0, prod(problem_size))) # initial design
println(size(x0))
p = 1.0 # penalty

# Young's modulus interpolation for compliance
# penalty1 = TopOpt.PowerPenalty(1.0) # take young modulus in each material 
# interp1 = MaterialInterpolation(Es, penalty1)

# # density interpolation for mass constraint
# penalty2 = TopOpt.PowerPenalty(1.0) #no penalty.
# interp2 = MaterialInterpolation(densities, penalty2)




# problem = PointLoadCantilever(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)
problem = HalfMBB(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)

#problem = HalfMBB(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)

solver = FEASolver(Direct, problem; xmin=xmin)

cheqfilter = DensityFilter(solver; rmin=rmin)
stress = TopOpt.von_mises_stress_function(solver)
comp = TopOpt.Compliance(solver)

function obj(x)
    # function constr(x)
    f = x[Int32(length(x) / 2)+1:end]
    v = x[1:Int32(length(x) / 2)]
    g = [x[1] for x in f2g.(f)]
    # minimize volume
    return sum(cheqfilter(PseudoDensities(v))) / length(x) - 0.1
end
function constr(x)
    # function obj(x)
    # compliance upper-bound
    f = x[Int32(length(x) / 2)+1:end]
    v = x[1:Int32(length(x) / 2)]
    return comp(cheqfilter(PseudoDensities(v .* (f2e(f))))) #- compliance_threshold
end

constr(x0)
gradient(constr, x0)
gradient(obj, x0)
m = TopOpt.Model(obj)
addvar!(m, vcat(zeros(length(x0) ÷ 2), 10 * ones(length(x0) ÷ 2)), vcat(ones(length(x0) ÷ 2), 100 * ones(length(x0) ÷ 2)))
Nonconvex.add_ineq_constraint!(m, constr)

options = MMAOptions(; maxiter=1000, tol=Tolerance(; kkt=1e-4, x=1e-4, f=1e-4))
TopOpt.setpenalty!(solver, p)
@time r = Nonconvex.optimize(
    m, MMA87(; dualoptimizer=ConjugateGradient()), x0; options=options
)




Amin = r.minimizer[1:Int32(length(r.minimizer) / 2)]
fmin = r.minimizer[Int32(length(r.minimizer) / 2)+1:end]
fmax = maximum(fmin)
fmin_n = fmin ./ fmax
@show obj(r.minimizer)
@show constr(r.minimizer)

@show maximum(stress(cheqfilter(PseudoDensities(Amin))))
topology = cheqfilter(PseudoDensities(Amin)).x

fig1 = visualize(problem; solver.u, topology=Amin, default_exagg_scale=0.0, scale_range=10.0)
Makie.display(fig1)

fig2 = visualize(problem; solver.u, topology=fmin_n, default_exagg_scale=0.0, scale_range=10.0)
Makie.display(fig2)


mapping = Array{Int64,2}(undef, ncells, 2)
for i in 1:160*40
    mapping[i, :] = [div(i, 160) + 1, mod(i, 160)]
end

f2 = Figure(resolution=(1000, 3000))
ax3 = Axis(f2[1, 1])
scatter!(ax3, mapping[:, 2], mapping[:, 1], color=y)


ax4, hm1 = heatmap(f2[2, 1], mapping[:, 2], mapping[:, 1], y)
ax5, hm2 = heatmap(f2[3, 1], mapping[:, 2], mapping[:, 1], y)

cbar1 = Colorbar(f2[1, 2])
cbar2 = Colorbar(f2[2, 2], hm1)
cbar3 = Colorbar(f2[3, 2], hm2)
f2



ax3.title = "penalty 3"
save("penalty3.png", f2)



cbar.ticks = ([-0.66, 0, 0.66], ["negative", "neutral", "positive"])
ax2.title = "comp: " * string(comp_lim)
ax2.title = "3mat"
using Colors, ColorSchemes
figure = (; resolution=(600, 400), font="CMU Serif")
axis = (; xlabel=L"x", ylabel=L"y", aspect=DataAspect())
#cmap = ColorScheme(range(colorant"red", colorant"green", length=3))
# this is another way to obtain a colormap, not used here, but try it.
mycmap = ColorScheme([RGB{Float64}(i, 1.5i, 2i) for i in [0.0, 0.25, 0.35, 0.5]])
fig, ax2, pltobj = heatmap(rand(-1:1, 20, 20);
    colormap=cgrad(mycmap, 3, categorical=true, rev=true), # cgrad and Symbol, mycmap
    axis=axis, figure=figure)
cbar = Colorbar(fig[1, 2], pltobj, label="Categories")
cbar.ticks = ([-0.66, 0, 0.66], ["negative", "neutral", "positive"])
f1
f2
optobj = obj(y)
text!("$optobj")
strval = split(string(comp_lim), ".")
# name = "aaafc_adjusted_multimat_compConts"*strval[1]*strval[2]*".png"
save(name, f2)

# end