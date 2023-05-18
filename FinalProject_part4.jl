

"""
Let's do Area optimizaiton first, get a very nice area, then do the fc optimization
"""


#### setup Topology Optimization for continuum #####
#get benchmark problem
# include("Benchmark1.jl")




# (min at 50)
compliance_threshold = 350 # maximum compliance
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
x0 = fill(1.0, ncells) , fill(100.0, ncellls) # initial design
p = 4.0 # penalty
V = 0.5 # maximum volume fraction

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

function obj(x)
    # function constr(x)
    fc = x[Int32(length(x) / 2)+1:end]
    den = x[1:Int32(length(x) / 2)]
    gwp = [x[1] for x in f2g.(fc)]
    # minimize volume
    return sum(cheqfilter(PseudoDensities(den.^3 .* gwp))) / length(x)*2 #- 0.4
end
function constr(x)
    # function obj(x)
    # compliance upper-bound
    f = x[Int32(length(x) / 2)+1:end]
    E = f2e(f)
    den = x[1:Int32(length(x) / 2)]
    return comp(cheqfilter(PseudoDensities(den .* E))) - compliance_threshold

m = Model(obj)
lb = zeros(length(x0)) , 10*ones(length(x0))
ub = ones(length(x0)) , 100*ones(length(x0))
addvar!(m, lb, ub)
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
    lines!(ax_A, [x1[1], x2[1]], [x1[2], x2[2]], color = :black, linewidth = Amin[i]*10)
end

f_E = Figure(resolution = (600 , 200))
ax_E = Axis(f_E[1, 1], xlabel = "x", ylabel = "y", title = "fc′")
#plot optimized truss structure
for i in eachindex(elements)
    x1 = node_points[elements[i][1]]
    x2 = node_points[elements[i][2]]
    lines!(ax_E, [x1[1], x2[1]], [x1[2], x2[2]], color = :black, linewidth = fmin[i]*10)
end


mapping = Array{Int64,2}(undef, ncells, 2)
for i in 1:160*40
    x = mod(i, 160)
    y = div(i, 160)+1
    if x == 0 
        x = 160
        y = y-1
    end
    mapping[i, :] = [x,y]
end

# ax3 = Axis(f2[1, 1])
# scatter!(ax3, mapping[:, 2], mapping[:, 1], color=y)



f2 = Figure(resolution = (600 , 200)) 
ax2, hm2 = heatmap(f2[1, 1], mapping[:, 1], mapping[:, 2], Amin)
ax2.title = "Area (desity)"
ax2.aspect = 3
cbar2 = Colorbar(f2[1,2], hm2)
f2



f3 = Figure(resolution=(600, 200))
ax3, hm1 = heatmap(f3[1, 1], mapping[:, 1], mapping[:, 2], fmin)
ax3.title = "fc′"
ax3.aspect = 3
cbar1 = Colorbar(f3[1,2], hm1)
f3

save("muti_fc"*string(compliance_threshold)*".png", f3)






# end

# # end