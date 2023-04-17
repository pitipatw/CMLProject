using GLMakie, GraphMakie
using GraphMakie.NetworkLayout
source = [1, 1, 2, 2, 3, 3];
target = [2, 3, 1, 3, 1, 2];
weight = [1.0, 0.5, 2.1, 2.3, 4, 4.1];

g = GNNGraph(source, target,weight)

g = smallgraph(:dodecahedral)
graphplot(g; layout=Stress(dim=3))