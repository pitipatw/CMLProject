#create GNN graph

using GraphNeuralNetworks

# From COO representation
source = [1, 1, 2, 2, 3, 3];
target = [2, 3, 1, 3, 1, 2];
weight = [1.0, 0.5, 2.1, 2.3, 4, 4.1];

g = GNNGraph(source, target,weight)

#then add later 

g.ndata.x = rand(Float32, 32 , 3) # 3 nodes, 32 features

#ndata dimension-> type , features , number of nodes
g = GNNGraph(source, target,weight, ndata = rand(Float32, 32 , 3))
g.ndata.x

#there are edge_data and edge_weight
get_edge_weight(g)

# or add right away
# You can have multiple feature arrays
g = rand_graph(10,  60, ndata = (; x=rand(Float32, 32, 10), y=rand(Float32, 10)))
#features
g.ndata.x



using Flux: DataLoader

data = [rand_graph(10, 30, ndata=rand(Float32, 3, 10)) for _ in 1:160]
gall = Flux.batch(data)

# gall is a GNNGraph containing many graphs
@assert gall.num_graphs == 160 
@assert gall.num_nodes == 1600   # 10 nodes x 160 graphs
@assert gall.num_edges == 9600  # 30 undirected edges x 2 directions x 160 graphs

# Let's create a mini-batch from gall
g23, _ = getgraph(gall, 2:3)
@assert g23.num_graphs == 2
@assert g23.num_nodes == 20   # 10 nodes x 160 graphs
@assert g23.num_edges == 120  # 30 undirected edges x 2 directions x 2 graphs x

# We can pass a GNNGraph to Flux's DataLoader
train_loader = DataLoader(gall, batchsize=16, shuffle=true)

using Flux: DataLoader

data = [rand_graph(10, 30, ndata= (;x = rand(Float32, 3, 10))) for _ in 1:320]
data = [rand_graph(10, 30, ndata= rand(Float32, 3, 10)) for _ in 1:320]
train_loader = DataLoader(data, batchsize=16, shuffle=true)


for g in train_loader
    @assert g.num_graphs == 16 ;
    @assert g.num_nodes == 160 ;
    println(g.ndata.x)
    @assert size(g.ndata.x) = (3,10)    
    # .....
end

train_loader = DataLoader(data, batchsize=16, shuffle=true)

using Flux
using Flux: DataLoader
using Flux: gpu

g_gpu = g |> gpu

data = [rand_graph(10, 30, ndata=rand(Float32, 3, 10)) for _ in 1:320]

train_loader = DataLoader(data, batchsize=16, shuffle=true)

for g in train_loader
    println("HI")
    @assert g.num_graphs == 16
    @assert g.num_nodes == 160
    println(size(g.ndata.x))# = (3, 160)    
    # .....
end

data = [rand_graph(10, 30, ndata=rand(Float32, 3, 10)) for _ in 1:320]

train_loader = DataLoader(data, batchsize=16, shuffle=true)

for g in train_loader
    println("HI")
    @assert g.num_graphs == 16
    @assert g.num_nodes == 160
    # @assert size(g.ndata.x) = (3, 160)    
    # .....
end

# Create a graph with a single feature array `x` associated to nodes
g = rand_graph(10,  60, ndata = (; x = rand(Float32, 32, 10)))

g.ndata.x  # access the features

# Equivalent definition passing directly the array
g = rand_graph(10,  60, ndata = rand(Float32, 32, 10))

g.ndata.x  # `:x` is the default name for node features

g.ndata.z = rand(Float32, 3, 10)  # add new feature array `z`

# For convenience, we can access the features through the shortcut
g.x 

# You can have multiple feature arrays
g = rand_graph(10,  60, ndata = (; x=rand(Float32, 32, 10), y=rand(Float32, 10)))

g.ndata.y, g.ndata.x   # or g.x, g.y

# Attach an array with edge features.
# Since `GNNGraph`s are directed, the number of edges
# will be double that of the original Graphs' undirected graph.
g = GNNGraph(erdos_renyi(10,  30), edata = rand(Float32, 60))
@assert g.num_edges == 60

g.edata.e  # or g.e

# If we pass only half of the edge features, they will be copied
# on the reversed edges.
g = GNNGraph(erdos_renyi(10,  30), edata = rand(Float32, 30))


# Create a new graph from previous one, inheriting edge data
# but replacing node data
g′ = GNNGraph(g, ndata =(; z = ones(Float32, 16, 10)))

g′.z
g′.e