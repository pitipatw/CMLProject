

"""
Let's do Area optimizaiton first, get a very nice area, then do the fc optimization

Is there a tradeoff? 
Is this better? 
"""


#### setup Topology Optimization for continuum #####
#get benchmark problem
# include("Benchmark1.jl")




# (min at 50)
compliance_threshold = 150 # maximum compliance
compliance_threshold = 46
# lc = [4000, 2000, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 49, 48, 47, 46]
# for i in lc
#     compliance_threshold  =  i
# 2D
ndim = 2
node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
    joinpath(@__DIR__, "tim_$(ndim)d.json")
)
ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
loads = load_cases["1"]
problem = TrussProblem(
    Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
)

xmin = 0.0001 # minimum density
x0 = vcat(fill(1.0, ncells) , fill(100.0, ncells)) # initial design
p = 4.0 # penalty
p = 1.0
solver = FEASolver(Direct, problem; xmin=xmin)
comp = TopOpt.Compliance(solver)

# ts = TopOpt.TrussStress(solver)
# ts(PseudoDensities(x0))

# function obj(x)
#     # minimize compliance
#     return comp(PseudoDensities(x))
# end
# function constr(x)
#     # volume fraction constraint
#     return sum(x) / length(x) - V
# end
L = Vector{Float64}(undef, ncells)
for i in 1:ncells
    L[i] = norm(node_points[elements[i][1]] - node_points[elements[i][2]])
end
function obj(x)
    # function constr(x)
    fc = x[Int32(length(x) / 2)+1:end]
    den = x[1:Int32(length(x) / 2)]
    gwp = [x[1] for x in f2g.(fc)]
    # minimize volume
    return sum(den.*gwp.*L)  #- 0.4
end
function constr(x)
    # function obj(x)
    # compliance upper-bound
    f = x[Int32(length(x) / 2)+1:end]
    E = f2e(f)
    A = x[1:Int32(length(x) / 2)]
    return comp(PseudoDensities(A .* E)) - compliance_threshold
end

m = TopOpt.Model(obj)
lb = vcat(zeros(length(x0)÷ 2) , 10*ones(length(x0)÷ 2))
ub =vcat(ones(length(x0)÷ 2) , 100*ones(length(x0)÷ 2))
addvar!(m, lb, ub)

@show constr(x0)
@show obj(x0)

Nonconvex.add_ineq_constraint!(m, constr)

options = MMAOptions(; maxiter=1000, tol=Tolerance(; kkt=1e-4, f=1e-4))
TopOpt.setpenalty!(solver, p)
@time r = Nonconvex.optimize(
    m, MMA87(; dualoptimizer=ConjugateGradient()), x0; options=options
)

@show obj(r.minimizer)
@show constr(r.minimizer)

Amin = r.minimizer[1:Int32(length(x0) / 2)]
fmin = r.minimizer[Int32(length(x0) / 2)+1:end]

f_A = Figure(resolution = (600 , 200))
ax_A = Axis(f_A[1, 1], xlabel = "x", ylabel = "y", title = "Area")
#plot optimized truss structure
for i in eachindex(elements)
    x1 = node_points[elements[i][1]]
    x2 = node_points[elements[i][2]]
    if Amin[i] > 0.0001
        lines!(ax_A, [x1[1], x2[1]], [x1[2], x2[2]], color = Amin[i], colorrange = (minimum(Amin):maximum(Amin)), linewidth = Amin[i]*10)
    end
end

Colorbar(f_A[1, 2], limits = (0,1), colormap = :viridis,
    flipaxis = false)



f_E = Figure(resolution = (600 , 200))
ax_E = Axis(f_E[1, 1], xlabel = "x", ylabel = "y", title = "fc′")
#plot optimized truss structure
for i in eachindex(elements)
    x1 = node_points[elements[i][1]]
    x2 = node_points[elements[i][2]]
    lines!(ax_E, [x1[1], x2[1]], [x1[2], x2[2]],colorrange=minimum(fmin):maximum(fmin),  color = fmin[i], colormap = :viridis, linewidth = 5)
end
Colorbar(f_E[1, 2], limits = (10,100), colormap = :viridis,
    flipaxis = false)
f_E
f_A

# ax3 = Axis(f2[1, 1])
# scatter!(ax3, mapping[:, 2], mapping[:, 1], color=y)
save("truss_A"*string(compliance_threshold)*".png", f_A)
save("truss_E"*string(compliance_threshold)*".png", f_E)

# save("muti_fc"*string(compliance_threshold)*".png", f3)

# text!(ax_E, "HI")




# end

# # end