# module ContComplianceDemo1

using TopOpt, LinearAlgebra, StatsFuns
using Makie, GLMakie
using TopOpt.TopOptProblems.Visualization: visualize

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 4.0 # filter radius
xmin = 0.0001 # minimum density
nx = 60
ny = 20
problem_size = (nx, ny)
V = 0.5 # maximum volume fraction
p = 4.0 # penalty
x0 = fill(V, prod(problem_size)) # initial design



problem = HalfMBB(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)

solver = FEASolver(Direct, problem; xmin=xmin)
cheqfilter = DensityFilter(solver; rmin=rmin)
comp = TopOpt.Compliance(solver)

function obj(x)
    # minimize compliance
    return comp(cheqfilter(PseudoDensities(x)))
end
function constr(x)
    # volume fraction constraint
    return sum(cheqfilter(PseudoDensities(x))) / length(x) - V
end

m = TopOpt.Model(obj)
addvar!(m, zeros(length(x0)), ones(length(x0)))
Nonconvex.add_ineq_constraint!(m, constr)

options = MMAOptions(; maxiter=1000, tol=Tolerance(; kkt=1e-4, x=1e-4, f=1e-4))
TopOpt.setpenalty!(solver, p)
# Method of Moving Asymptotes
@time r = Nonconvex.optimize(m, MMA87(), x0; options=options)

@show obj(r.minimizer)
@show constr(r.minimizer)
topology = cheqfilter(PseudoDensities(r.minimizer)).x
fig = visualize(problem; solver.u,
    topology = topology, default_exagg_scale=0.0, scale_range=10.0)
Makie.display(fig)

## Pitipat edited after this line 
ncells = nx*ny
y = r.minimizer
mapping = Array{Int64,2}(undef, ncells, 2)
for i in 1:nx*ny
    mapping[i, :] = [div(i, nx) + 1, mod(i, nx)]
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

