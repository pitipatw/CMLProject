using TopOpt
using Makie, GLMakie


Base.@kwdef mutable struct Truss
    nk::Int32
    nm::Int32
    init_node::Dict{Int32,Vector{Int32}} # Matrix{Float32}
    connectivity::Dict{Int32,Vector{Int32}} #Matrix{Int32}
    pin_nodes::Dict{Int32,Vector{Int32}} #Matrix{Int32}
    nsize:: Int32 # 10.0
    lsize:: Int32 # 4.0
    spsize:: Int32 # = 17.0
    support::Dict{Int32,Vector{Int32}} #
    node::Dict{Int32,Vector{Int32}} #
    material::Dict{Int32,Vector{Int32}} #
    done::Bool # = false
    steps::Int32 # = 0
    total_reward::Float32 # 0.0
    existence::Dict{Int32,Int32} # # ones(nm,1)
    exist_member_i:: Dict{Int32,Int32} # # = 1:nm
    section:: Dict{Int32,Int32} # #ones(nm,1)
    #connectivity is basically elements from TopOpt jl
    # rewrite function Truss
    function Truss(init_node::Dict{Int32,Vector{Int32}}, connectivity::Dict{Int32,Vector{Int32}}, pin_nodes::Dict{Int32,Vector{Int32}}, nsize:: Int32, lsize:: Int32, spsize:: Int32, support:: Dict{Int32,Vector{Int32}})
        nk = length(init_node)
        nm = length(connectivity)
        done = false
        steps = 0 
        total_reward = 0.0
        existence = Dict{Int32,Int32}()
        exist_member_i = Dict{Int32,Int32}()
        section = Dict{Int32,Int32}()
        node = copy(init_node)
        new(nk, nm, init_node, connectivity, pin_nodes, nsize, lsize, spsize, support, node, material, done, steps, total_reward, existence, exist_member_i, section)
    end
end
# 2D


# function

ndim = 2
node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
    joinpath(@__DIR__, "tim_$(ndim)d.json")
);


begin
#need to generate these
node_points
elements
fixities
load_cases
mats
crosssecs
###
end

ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
loads = load_cases["0"]

problem_original = TrussProblem(
    Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
);

#TopOptjl -> hayshi's format 




xmin = 0.0001
solver = FEASolver(Direct, problem_original; xmin=xmin)
ts = TrussStress(solver)

areas = Array{Float32}(undef, length(crosssecs))
for i in eachindex(crosssecs)
    areas[i] = crosssecs[i].A
end

σ = ts(PseudoDensities(areas))

# pick an element to remove (set value to 0.0001)
crosssecs2 = deepcopy(crosssecs)
i = 2
crosssecs2[i] = TrussFEACrossSec(0.0001)
problem_mod = TrussProblem(
    Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs2
);

solver2 = FEASolver(Direct, problem_mod; xmin=xmin)
ts2 = TrussStress(solver2)
areas2 = Array{Float32}(undef, length(crosssecs))
for i in eachindex(crosssecs2)
    areas2[i] = crosssecs2[i].A
end

σ2 = ts(PseudoDensities(areas2))
σ
#plot structure and stress as red and blue color 

fig = Figure(resolution = (800, 600))
ax = Axis(fig[1, 1])
for i in eachindex(elements)
    e = elements[i]
    n1, n2 = node_points[e[1]], node_points[e[2]]
    x1, x2 = n1[1], n2[1]
    y1, y2 = n1[2], n2[2]
    if i == 1
        lines!(ax, [x1, x2], [y1, y2], linewidth = 100*abs(σ2[i]), color = :green)
    elseif σ[i] > 0
        lines!(ax, [x1, x2], [y1, y2], linewidth = 100*abs(σ2[i]), color = :blue)
    else
        lines!(ax, [x1, x2], [y1, y2], linewidth = 100*abs(σ2[i]), color = :red)
    end
end

