# node state data vk for nodes. 
# index 
# 1 : 1 if pin-supported, else 0 
# 2 Load intensity [kN] at the node in x direction (load case 1)
# 3 Load intensity [kN] at the node in y direction (load case 1)
# .
# .
# .
# 2nL Load intensity [kN] at the node in x direction (load case nL)
# 2nL+1 Load intensity [kN] at the node in y direction (load case nL)

# member state data wi for members
# index
# 1 cos α : α: the angle of the member with respect to positive x direction
# 2 sin α : α: the angle of the member with respect to positive x direction
# 3 Member length
# 1 if remained, 0 if removed
# Stress safety ratio (load case 1) 
# .
# .
# .
# nL+4 Stress safety ratio (load case nL)
using ReinforcementLearning
using Flux
using Asap
include("GS.jl")
#pre processing

σbar =   200.0 #N/mm2
δbar = 100.0 #*(max())
E    = 2.0e5 #N/mm2
#initial cross section area 
A0 = 1000.0 # mm2

node_points = Dict{Int64,Vector{Float64}}()
node_counter = 0
nx = 5
ny = 5
nz = 1

for i in 1:nz
    for j in 1:ny
        for k in 1:nx
            global node_counter += 1
            node_points[node_counter] = [j-1.0,k-1.0,i-1.0]
        end
    end
end

for i in sort(collect(keys(node_points)))
    println(i)
    println(node_points[i])
end

GS1 = getGS(node_points, 1.5)

#using Asap format here
#random 2 numbers from 1 to 5 
# node_points = Dict{Int64,Vector{Float64}}()
n1 = rand(1:2) 
n2 = rand(4:5)  
println(n1," & ",n2)

# for 5x5 the right most dof are from dof = 42 44 46 48 50
load_point = rand( [ 42 44 46 48 50 ] )
load_mag = 1.0 #kN

nodes = Vector{Asap.TrussNode}()
     #Vector{Any}(undef, length(node_points))
for i in range(1,25)
    if i == n1 || i == n2
        push!(nodes,Asap.TrussNode(node_points[i], :pinned))
    else
        push!(nodes,Asap.TrussNode(node_points[i], :zfixed))
    end
end


load1 = [Asap.NodeForce(nodes[Int(load_point/2)], [ 0., -load_mag, 0.])]

sec = Asap.TrussSection( A0, E)

GS1 = getGS(node_points, 1.5)
elements = Vector{Asap.TrussElement}()

for i in eachindex(GS1)
    push!(elements,Asap.TrussElement(nodes ,[GS1[i][1], GS1[i][2] ]  ,sec))
end

#assemble model
model = Asap.TrussModel(nodes, elements, load1)

#solve model
Asap.solve!(model)

#extract information
println(model.u)
println(elements[1].forces)
println(nodes[1].reaction)

σ = []
for i in eachindex(elements)
    push!(σ, elements[i].forces[1]/elements[i].section.A)
end

println(maximum(σ))
#Done structural calculation.

#training episodes 
#number of episodes
nE = 5000
γ = 0.99



# elements[1].section.A = 1e-6

# from model, elements and node, compute state.

# put model, elements, nodes into v andd w form 

# v = Matrix{Float64}(undef, 1, 3)
v = []
w = [] # Matrix{Float64}()
for i in eachindex(nodes)
    println(i)
    vi = []
    if sum(nodes[1].dof) == 0
        push!(vi, 1)
    else
        push!(vi, 0)
    end

    push!(vi , nodes[i].reaction[1])
    push!(vi , nodes[i].reaction[2])
    #these 2 lines are for the load case 2, which mean, I have to do another model.
    # push!(vi , node[i].reaction[1])
    # push!(vi , node[i].reaction[2])
    #has to be concat column wise
    if i == 1 
        v = vi
    else
        v = hcat(v,vi)
    end
    # push!(v, vi)
end

exist_list = ones(length(elements))
for i in eachindex(elements)
    wi = []
    push!(wi, cos(elements[i].Ψ))
    push!(wi, sin(elements[i].Ψ))
    push!(wi, elements[i].length)
    push!(wi, exist_list[i])
    push!(wi, elements[i].forces[1]/elements[i].section.A/σbar)
    #push!(wi, elements[i].forces[2]/elements[i].section.A/σbar)
    # push!(w, wi)
    if i == 1
        w = wi
    else
        w = hcat(w,wi)
    end
end

#now we have w and v


#generate Ground Stucture based on set of node points
# use Robar's function

# speciy upper-bound values for stress and displacement

# specify graph embedding class
# nodes state

nL = 1 #number of load cases
nd = 2 * length(nodes) # number of total dof
nm = length(elements) # number of members
nf = 100 # size of the feature vector of a member 
#nf obtain through trial and error.







#reward function 

function reward( i::Int32, σbar::Float32 , δbar::Float32,model::Asap.Model, w::Array{Float32,2})
    #check stress constrain
    # i is the element index that was removed. 
  
    #check displacement constrain
    #check max stress and max displacement
    maxσ = maximum(w[5,:])
    maxδ = maximum(model.u)

    if maxσ > σbar || maxδ > δbar
        return -1
    else
        Le = w[3,i]
        return Le * (1 - maxσ/σbar)
    end
end

ϵ = 0.1
a = EpsilonGreedyExplorer(ϵ)


#initialize the learning parameters
θ1 = Matrix{Float64}(undef, nf, nL + 4)
θ2 = Matrix{Float64}(undef, nf, nf)
θ3 = Matrix{Float64}(undef, nf, 2*nL+1)
θ4 = Matrix{Float64}(undef, nf, nf)
θ5 = Matrix{Float64}(undef, nf, nf)
θ6 = Matrix{Float64}(undef, nf, nf)
θ7 = Matrix{Float64}(undef, 2*nf,1)
θ8 = Matrix{Float64}(undef, nf, nf)
θ9 = Matrix{Float64}(undef, nf, nf)

#policy
#policy is a function that takes in a state and outputs an action
#policy will input a state and i and get the hiest value of reward. 

#we are using epsilon-greedy policy
#we will use a random number generator to decide if we will explore or exploit
#if the random number is less than epsilon, we will explore
#if the random number is greater than epsilon, we will exploit
function policy(state, i)
    if rand() < epsilon
        #explore
        return rand(1:2)
    else
        #exploit
        return argmax([Qπ(state, i)])
    end
end



    #action value estimate by graph embedding, these are learnable parameter

    # μi is a feature vector of a member i for i = 1 to nm 

μ̂  = Matrix{Float64}(undef, nf, nm)

#should embed this into graph now 
(xi, xj, e) -> xi .+ xj
function updateMember()
    #for edge (member) i
    h1 = θ1 * w[:,i]
    h2 = θ2* sum( NNlib.relu.( θ3 *xi , θ3*xj ) )
    h3 = θ4 * μ


    return NNlib.relu.( h1 + h2 + h3 + h4) 
end



#look at each i 
for i in range(nm) 
    μi_old = zeros(nf)
    h1 = θ1 * w[:,i]
    h2 = θ2* sum( NNlib.relu.( θ3 *xi , θ3*xj ) )
    for _ in range(4)

        h3 = θ4 * μi_old
        h4 = 0.0 #initiate
        sum_outer = 0.0
        for j in range(2)
            sum_inner = 0.0
            #j is the end of the member i
            #phi i j is the set of indices of members connecting to j th  end of member i. And does not include i itself)
            for k in eachindex(node_conected to the J end)
                element_number = node_points[elements[i][j]]
                if element_number != i
                    sum_inner += μ[:,element_number]
                end
                sum_inner= NNlib.relu( sum_inner )
            
            end
            sum_outer += sum_inner
        end
        h4 = θ5 * sum( NNlib.relu.( θ6 *μi_old + θ6*) )
        ui_old = ui_new

    end
    μ̂[:,i] = ui_new

# phi i j is the set of indices of members connecting to j th  end of member i. And does not include i itself) 


using t = 0 to 4 (iterate 4 times)

later, use that to create û that has concate column wise of all u 
û = Matrix{Float64}(undef, nf, nm)

for i in range(nm) 
    û[:,i] = ui
end
#introducing 2 more parameters

Q = θ7' ( NNlib.relu( θ8 * sum i = 1 to nm of [u(i) ; θ9*ui))


Θ = [θ1 ; θ2 ; θ3 ; θ4 ; θ5 ; θ6 ; θ7 ; θ8 ; θ9]

#Q learning using the embeded features


Q(s,a) = Q(s,a) + α * (r(s') + γ * max(Q(s',a)) - Q(s,a))

# the problem becomes 
minimize F(Θ) = ( r(s') + γ * max(Q(s',a; Θ̃)) - Q(s,a; Θ)) ^2

#start training 

# there should be a truss analysis functoin to get displacement of each dof and stress of each memeber
function truss_analysis(GS, bcs , F , A, vk, wi)
    GS is a connectivity of the each nodes 
    bcs is the boundary condition of each node
    F is the load of each dof
    A is the cross section area of each member, which is a constant 
    vk and wi are described above.

    #GS is the ground structure
    #A is the cross section area of each member
    #vk is the state of each node
    #wi is the state of each member

    #return the displacement of each dof and stress of each member
    return nothing
end


function train()
    #initialize cross section area
    all A = A0 

    #randomize the boundary condition and loads
    2 pins are chosen from the left most nodes
    2 loading points with load value of 1kN are chosen from the right most nodes. 

    #compute state s = {v , w} 

    state = getState(GS, bcs, F, A)
    Q = getQ(state) 
end


function getState(GS, bcs, F, A)
    #compute the state of the system
    #GS is the ground structure
    #A is the cross section area of each member
    #bcs is the boundary condition of each node
    #F is the load of each dof

    #return the state of the system
    return nothing
end

function getQ(state, Θ) 
    #compute Q for the remaining memebers
    for i in eachindex(m) 
        if area != 0 
            compute.
        choose a member to remove by epsilon greedy policy
        compute the reward
        update Q
    end

    #update Θ with RMSprop

    return Θ
end


function TEST()
    #set the boundary condition and loads
    2 pins are chosen from the left most nodes
    2 loading points with load value of 1kN are chosen from the right most nodes. 

    #compute state s = {v , w} 

    state = getState(GS, bcs, F, A)
    Q = getQ(state) 
end