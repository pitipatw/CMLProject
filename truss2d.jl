# 2D
ndim = 2
node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
    joinpath(@__DIR__, "tim_$(ndim)d.json")
);

#need to generate these
node_points
elements
fixities
load_cases
mats
crosssecs
###


ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
loads = load_cases["0"]
problem = TrussProblem(
    Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
);

solver = FEASolver(Direct, problem; xmin=xmin)


ts = TrussStress(solver)
σ = ts(PseudoDensities(crosssecs))




