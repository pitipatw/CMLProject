function getQ(s::State , a ::Action, Θ::Vector{Float64})
    #compute Q for the remaining memebers
    r_list = zeros(length(a))
    v,w = s
    for i in eachindex(a)
        m_id = a[i]
        le = w[3,m_id]
        #set the area of element m_id to Amin 

        #calculate the stress from the new state (structure) 
        #this is where topopt jl comes 

        #get the max stress / cap stress

        #calculate the reward
        r = Le*( 1 - max(stress_raio))
        r_list[i] = r

        #     compute.
        # choose a member to remove by epsilon greedy policy
        # compute the reward
        # update Q
    end

    #update Θ with RMSprop

    return Θ
end


function updateEdges()