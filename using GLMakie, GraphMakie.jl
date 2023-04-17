using GLMakie, GraphMakie
using GraphMakie.NetworkLayout
source = [1, 1, 2, 2, 3];
target = [2, 3, 1, 3, 1];
weight = [1.0, 0.5, 2.1, 2.3, 4]; # distance between the nodes

g = GNNGraph(source, target,weight)

g = smallgraph(:dodecahedral)
graphplot(g)#; layout=Stress(dim=3))



node_points =Dict(1 =>[0.0, 0.0,0.0] , 2 =>[1.0, 0.0,0.0],
3 => [0.0, 1.0,0.0], 4 => [0.0, 0.0,1.0],   5 => [1.0, 1.0,0.0])

elements = Dict(1 => [1,2], 2 => [1,3], 3 => [1,4], 4 => [1,5], 5 => [2,3], 6 => [2,4], 7 => [2,5], 8 => [3,4], 9 => [3,5], 10 => [4,5])
#let's create graph g
source = Vector{Int64}()
target = Vector{Int64}()
weight = Vector{Float64}()



for (k,v) in elements
    push!(source, elements[k][1])
    push!(target, elements[k][2])

    @show x1 = node_points[elements[k][1]][1]
    @show y1 = node_points[elements[k][1]][2]
    z1 = node_points[elements[k][1]][3]
    x2 = node_points[elements[k][2]][1]
    y2 = node_points[elements[k][2]][2]
    z2 = node_points[elements[k][2]][3]
    distance = sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
    push!(weight, distance)
end

g = GNNGraph(source, target,weight)
graphplot(g, node_attr = (nlabels,collect(keys(node_points))))
    #elabels = weight,)