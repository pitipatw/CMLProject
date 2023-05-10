

"""
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
"""


#### setup Topology Optimization for continuum #####
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