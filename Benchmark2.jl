using TopOpt, LinearAlgebra, StatsFuns
#using Makie, GLMakie
#using TopOpt.TrussTopOptProblems.TrussVisualization: visualize

println("B2 STARTS")
# 2D
ndim = 2
node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
    joinpath(@__DIR__, "tim_2d.json")
)
ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
loads = load_cases["0"]
problem = TrussProblem(
    Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
)

xmin = 0.0001 # minimum density
x0 = fill(2.0, ncells) # initial design
p = 1.0 # penalty
compliance_threshold = 2 # maximum compliance
solver = FEASolver(Direct, problem; xmin=xmin)
comp = TopOpt.Compliance(solver)


L = Vector{Float64}(undef, ncells)
for i in 1:ncells
    L[i] = norm(node_points[elements[i][1]] - node_points[elements[i][2]])
end

function obj(x)
    # minimize volume
    return sum(x./L)
end
function constr(x)
    # compliance upper-bound
    return comp(PseudoDensities(x)) - compliance_threshold
end

constr(x0)

m = Model(obj)
addvar!(m, zeros(length(x0)), 3*ones(length(x0)))
Nonconvex.add_ineq_constraint!(m, constr)

options = MMAOptions(; maxiter=1000, tol=Tolerance(; kkt=1e-4, f=1e-4))
TopOpt.setpenalty!(solver, p)
@time r = Nonconvex.optimize(
    m, MMA87(; dualoptimizer=ConjugateGradient()), x0; options=options
)

@show obj(r.minimizer)
@show constr(r.minimizer)
#fig = visualize(
#    problem; solver.u, topology = r.minimizer,
#    default_exagg_scale=0.0
#)
#Makie.display(fig)

println("#"^50)
println("B2 report:")
println("B2 has compliance threshold: ", compliance_threshold)
println("B2 has compliance: ", comp(PseudoDensities(r.minimizer)))
println("B2 has volume fraction: ", sum(r.minimizer) / length(r.minimizer))
println("B2 has penalty: ", p)
println("B2 has objective: ", obj(r.minimizer))
println("END of Benchmark2.jl")
println("#"^50)


@show obj(r.minimizer)
@show constr(r.minimizer)

Amin = r.minimizer

f_b2 = Figure(resolution = (600 , 200))
ax_b2 = Axis(f_b2[1, 1], xlabel = "x", ylabel = "y", title = "Area")
#plot optimized truss structure
for i in eachindex(elements)
    x1 = node_points[elements[i][1]]
    x2 = node_points[elements[i][2]]
    if Amin[i] > 0.0001
        lines!(ax_b2, [x1[1], x2[1]], [x1[2], x2[2]], color = Amin[i], colorrange = (0:0.1:maximum(Amin)), linewidth = 10*Amin[i])
    end
end

Colorbar(f_b2[1, 2], limits = (0,1), colormap = :viridis,
    flipaxis = false)

f_b2

save("b2_comp_is_"*string(compliance_threshold)*".png", f_b2)
println("SAVE COMPLETE")
println("b2 ENDS")
