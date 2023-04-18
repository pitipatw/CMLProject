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

#pre processing

#generate Ground Stucture based on set of node points
# use Robar's function
GS = nothing #elements

# speciy upper-bound values for stress and displacement
σ̄ = nothing
δ̄ = nothing

# specify graph embedding class
# nodes state

nL = 1 #number of load cases
nd = 2* (numberofnodes) # number of total dof
nm = numberofmembers # number of members
nf = xxx # size of the feature vector of a member 
#nf obtain through trial and error.



#initial cross section area 
A0 = 1000.0 # mm2



#reward function 
function reward( Le, σ , δ,)
    #check stress constrain
    

    #check displacement constrain

    if those constrain are violated σ/ σ̄ > 1 or δ/ δ̄ > 1
        return -1
    else 
        return Le * (1- max( σ/σ̂ , δ/δ̂))

    end

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
            return argmax([Qπ(state, i)
        end
    end


    #action value estimate by graph embedding, these are learnable parameters.
    θ1 = Matrix{Float64}(undef, nf, nL + 4)
    θ2 = Matrix{Float64}(undef, nf, nf)
    θ3 = Matrix{Float64}(undef, nf, 2*nL+1)
    θ4 = Matrix{Float64}(undef, nf, nf)
    θ5 = Matrix{Float64}(undef, nf, nf)
    θ6 = Matrix{Float64}(undef, nf, nf)

    # μi is a feature vector of a member i for i = 1 to nm 

μ̂ = Matrix{Float64}(undef, nf, nm)

h1 = θ1 * w
h2 = θ2 * sum(j = 1 to 2 ) ReLU(θ3 * v[i,j])
h3 = θ4 * mu(i) at t 
h4 = θ5 * sum(j = 1 to 2 ) ReLU(θ6 *  sum( mu k from k = set of members connected to i at t)
phi i j is the set of indices of members connecting to j th  end of member i. And does not include i itself) 


using t = 0 to 4 (iterate 4 times)

later, use that to create û that has concate column wise of all u 
û = Matrix{Float64}(undef, nf, nm)

for i in range(nm) 
    û[:,i] = ui
end
#introducing 2 more parameters
θ7 = Matrix{Float64}(undef, 2*nf)
θ8 = Matrix{Float64}(undef, nf, nf)
θ9 = Matrix{Float64}(undef, nf, nf)

Q = θ7' ( ReLU( θ8 * sum i = 1 to nm of [u(i) ; θ9*ui))


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