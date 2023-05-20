using TopOpt, LinearAlgebra, StatsFuns
using Makie, GLMakie
# using TopOpt.TopOptProblems.Visualization: visualize

println("B1 STARTS")
E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 4.0 # filter radius
xmin = 0.0001 # minimum density
nx = 160
ny = 40
ncells = nx*ny
problem_size = (160, 40)
x0 = fill(1.0, prod(problem_size)) # initial design
p = 4.0 # penalty
compliance_threshold = 800 # maximum compliance

# problem = PointLoadCantilever(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)
problem = HalfMBB(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)

solver = FEASolver(Direct, problem; xmin=xmin)

cheqfilter = DensityFilter(solver; rmin=rmin)
stress = TopOpt.von_mises_stress_function(solver)
comp = TopOpt.Compliance(solver)

function obj(x)
    # minimize volume
    return sum(cheqfilter(PseudoDensities(x))) / length(x)
end
function constr(x)
    # compliance upper-bound
    return comp(cheqfilter(PseudoDensities(x))) - compliance_threshold
end

m = TopOpt.Model(obj)
addvar!(m, zeros(length(x0)), ones(length(x0)))
Nonconvex.add_ineq_constraint!(m, constr)

options = MMAOptions(; maxiter=1000, tol=Tolerance(; kkt=1e-4, x=1e-4, f=1e-4))
TopOpt.setpenalty!(solver, p)
@time r = Nonconvex.optimize(
    m, MMA87(; dualoptimizer=ConjugateGradient()), x0; options=options
)

@show obj(r.minimizer)
@show constr(r.minimizer)
@show maximum(stress(cheqfilter(PseudoDensities(r.minimizer))))
topology = cheqfilter(PseudoDensities(r.minimizer)).x
# fig = visualize(problem; solver.u,
#     topology = topology, default_exagg_scale=0.0, scale_range=10.0)
# Makie.display(fig)


println("#"^50)
println("B1 report:")
println("B1 has compliance threshold: ", compliance_threshold)
println("B1 has compliance: ", comp(PseudoDensities(r.minimizer)))
println("B1 has volume fraction: ", sum(r.minimizer) / length(r.minimizer))
println("B1 embodied carbon: ",sum(r.minimizer) * 0.446 )
println("B1 has penalty: ", p)
println("B1 has objective: ", obj(r.minimizer)*0.446)
println("END of Benchmark1.jl")
println("#"^50)




A_b1 = r.minimizer
mapping = Array{Int64,2}(undef, ncells, 2)
for i in 1:nx*ny
    x = mod(i, nx)
    y = div(i, nx)+1
    if x == 0 
        x = nx
        y = y-1
    end
    mapping[i, :] = [x,y]
end

f_b1 = Figure(resolution = (800, 600))
ax_b1, hm_b1 = heatmap(f_b1[1, 1], mapping[:, 1], mapping[:, 2], A_b1)

ax_b1.title = "Area (desity)"
ax_b1.aspect = 3
cbar_b1 = Colorbar(f_b1[1,2], hm_b1)
f_b1
save("b1_com_is_"*string(compliance_threshold)*".png", f_b1)
println("SAVE COMPLETE")
println("B1 ENDS")
# # end
