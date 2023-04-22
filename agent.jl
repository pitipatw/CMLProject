#Python code to Julia
# import os
# import numpy as np

# import random

# import copy
# import torch
# torch.manual_seed(0)
# from collections import deque
# from dataclasses import dataclass
# import pickle
# import zlib

using Flux
using Flux.Optimise
using DataStructures
#using CuArrays
using GraphNeuralNetworks
using LinearAlgebra
using Random, Distributions


#Initialize random
# np.random.seed(0)

# random.seed(0)

### User specified parameters ###
INIT_MEAN = 0.0 ## mean of initial training parameters
INIT_STD = 0.05 ## standard deviation of initial training parameters
TARGET_UPDATE_FREQ = 100
USE_BIAS = False
#################################

# @dataclass
struct Experience
	c::Matrix{Float32} #np.ndarray # connectivity
	v::Matrix{Float32} #torch.Tensor
	w::Matrix{Float32} #torch.Tensor
	action ::Int32 # np.int32
	reward ::Float32 # np.float32
	c_next ::Matrix{Float32} # np.ndarray
	v_next ::Matrix{Float32} #torch.Tensor
	w_next ::Matrix{Float32} #torch.Tensor
	done :: Bool
	infeasible_action:: Matrix{Float32} #np.ndarray
end

# @dataclass
struct Temp_Experience #class Temp_Experience:
	c::Matrix{Float32} #np.ndarray # connectivity
	v::Matrix{Float32} # torch.Tensor
	w::Matrix{Float32} # torch.Tensor
	action::Int32 # np.int32
	reward::Float32# np.float32
end

#class NN(torch.nn.Module):
Base.@kwdef mutable struct NN
		# def __init__(self,n_node_inputs,n_edge_inputs,n_feature_outputs,n_action_types,batch_size,use_gpu):
	# 	super(NN,self).__init__()
	n_node_inputs::Int32
	n_edge_inputs::Int32
	n_feature_outputs::Int32
	n_action_types::Int32
	batch_size::Int32
	use_gpu::Bool
	l1_1::Flux.Dense
	l1_2::Flux.Dense
	l1_3::Flux.Dense
	l1_4::Flux.Dense
	l1_5::Flux.Dense
	l1_6::Flux.Dense
	l2_1::Flux.Dense
	ActivationF::Function
	device::Any

	function NN(;n_node_inputs::Int32,n_edge_inputs::Int32,n_feature_outputs::Int32,n_action_types::Int32,batch_size::Int32,use_gpu::Bool)
		l1_1 = Flux.Dense(n_edge_inputs,n_feature_outputs ,bias = false ;init = Flux.glorot_normal)    # self.l1_1 = torch.nn.Linear(n_edge_inputs,n_feature_outputs,False)
		l1_2 = Flux.Dense(n_feature_outputs,n_feature_outputs ,bias = false ;init = Flux.glorot_normal)# self.l1_2 = torch.nn.Linear(n_feature_outputs,n_feature_outputs)
		l1_3 = Flux.Dense(n_node_inputs,n_feature_outputs ,bias = false;init = Flux.glorot_normal) 	   # self.l1_3 = torch.nn.Linear(n_node_inputs,n_feature_outputs)
		l1_4 = Flux.Dense(n_feature_outputs,n_feature_outputs ,bias = false;init = Flux.glorot_normal) # self.l1_4 = torch.nn.Linear(n_feature_outputs,n_feature_outputs)
		l1_5 = Flux.Dense(n_feature_outputs,n_feature_outputs ,bias = false;init = Flux.glorot_normal) # self.l1_5 = torch.nn.Linear(n_feature_outputs,n_feature_outputs)
		l1_6 = Flux.Dense(n_feature_outputs,n_feature_outputs ,bias = false;init = Flux.glorot_normal) # self.l1_6 = torch.nn.Linear(n_feature_outputs,n_feature_outputs)
		l2_1 = Flux.Dense(n_feature_outputs,n_action_types ,bias = false;init = Flux.glorot_normal) # self.l2_1 = torch.nn.Linear(n_feature_outputs,n_action_types)
		ActivationF = leakyrelu
		if use_gpu
			#self.to('cuda') not sure about this one
			device = Flux.gpu # self.device = torch.device('cuda')
		else
			#self.to('cpu')
			device = Flux.cpu # self.device = torch.device('cpu')
		end

		new(n_node_inputs,n_edge_inputs,n_feature_outputs,n_action_types,batch_size,use_gpu,l1_1,l1_2,l1_3,l1_4,l1_5,l1_6,l2_1,ActivationF)
	end
		# self.l2_2 = torch.nn.Linear(n_feature_outputs,n_feature_outputs,bias=USE_BIAS)
		# self.l2_3 = torch.nn.Linear(n_feature_outputs,n_feature_outputs,bias=USE_BIAS)
	# batch_size = batch_size
		# self.batch_size = batch_size
	#  self.ActivationF = torch.nn.LeakyReLU(0.2)

	# Initialize_weight!() #this is another function 

	# self.n_feature_outputs = n_feature_outputs


end

# the rest of the function down here can be put outside the type definition.
function Connectivity(self::NN,connectivity::Matrix{Int32},n_nodes::Int32)
	
	# connectivity[n_edges,2]
	n_edges = size(connectivity)[1] # n_edges = connectivity.shape[0]
	order = 1:n_edges # order = np.arange(n_edges)
	adjacency = zeros(Float32,n_nodes,n_nodes) |> self.device # adjacency = torch.zeros(n_nodes,n_nodes,dtype=torch.float32,device=self.device,requires_grad=False)
	incidence = zeros(Float32,n_nodes,n_edges) |> self.device # incidence = torch.zeros(n_nodes,n_edges,dtype=torch.float32,device=self.device,requires_grad=False)

	for i in 1:2
		adjacency[connectivity[:,i],connectivity[:,(i+1)%2]] = 1
	end

	incidence[connectivity[:,1],order] = -1
	incidence[connectivity[:,2],order] = 1

	incidence_A = abs.(incidence) # incidence_A = torch.abs(incidence)#.to_sparse()
	incidence_A = torch.abs(incidence)#.to_sparse()
	incidence_1 = (incidence.==-1) |> Float32 |> self.device # incidence_1 = (incidence==-1).type(torch.float32)
	incidence_2 = (incidence.==1)  |> Float32 |> self.device # incidence_2 = (incidence==1).type(torch.float32)

	return incidence_A,incidence_1,incidence_2,adjacency
end

	
# function Initialize_weight!(self)
# 	# initialize weights using normalization with the predefined std and mean

# end

	# function Output_params(self)
	# 	for name,m in self.named_modules()
	# 		if isinstance(m,torch.nn.Linear)
	# 			print(name)
	# 			np.savetxt(f"agent_params/{name}_w.npy",m.weight.detach().to('cpu').numpy())
	# 			if m.bias != nothing
	# 				np.savetxt(f"agent_params/{name}_b.npy",m.bias.detach().to('cpu').numpy())
	# 			end
	# 		end
	# 	end
	# end

function getμ(self::NN,v::Matrix{Int32},μ::Matrix{Int32} ,w::Matrix{Int32},incidence_A::Matrix{Int32},incidence_1::Matrix{Int32},incidence_2::Matrix{Int32},adjacency::Matrix{Int32}, μ_iter::Int32)
	
	# v (array[n_nodes,n_node_features])
	# mu(array[n_edges,n_edge_out_features])
	# w (array[n_edges,n_edge_in_features])

	if μ_iter == 0
		h1   = self.l1_1(w)
		h2_0 = self.ActivationF(self.l1_3(v))
		h2   = self.l1_2(incidence_A' * h2_0)
		μ    = self.ActivationF(h1 + h2)

	else
		h3   = self.l1_6(μ)
		h4_0 = incidence_A * μ

		n_connect_edges_1 = clamp( repeat(sum(adjacency' * incidence_1,dims=1), outer = (self.n_feature_outputs,1))'-1,1,Inf)
		n_connect_edges_2 = clamp( repeat(sum(adjacency' * incidence_2,dims=1), outer = (self.n_feature_outputs,1))'-1,1,Inf)
		# n_connect_edges_1 = torch.clip(torch.sum(torch.mm(adjacency.T,incidence_1),axis=0).repeat(self.n_feature_outputs,1).T-1,1)
		# n_connect_edges_2 = torch.clip(torch.sum(torch.mm(adjacency.T,incidence_2),axis=0).repeat(self.n_feature_outputs,1).T-1,1)
		h4_1 = (self.l1_4(incidence_1' * h4_0) - μ ) /n_connect_edges_1
		h4_2 = (self.l1_4(incidence_2' * h4_0) - μ ) /n_connect_edges_2
		h4 = self.l1_5(self.ActivationF(h4_1) + self.ActivationF(h4_2))
		μ = self.ActivationF(h3 + h4)
		# h4_1 = self.l1_4.forward(torch.mm(incidence_1.T,h4_0)-mu)/n_connect_edges_1
		# h4_2 = self.l1_4.forward(torch.mm(incidence_2.T,h4_0)-mu)/n_connect_edges_2
		# h4 = self.l1_5.forward(self.ActivationF(h4_1)+self.ActivationF(h4_2))
		# mu = self.ActivationF(h3+h4)
	end

	return μ
end
		
function getQ(self::NN,μ::Matrix{Int32},n_edges::Int32)
	
	if typeof(n_edges) == Int # normal operation
		μ_sum = sum(mu,dims=1)
		μ_sum = repeat(μ_sum,outer = (n_edges, 1))
	else # for mini-batch training
		μ_sum = zeros(Float32, (n_edges[-1],self.n_feature_outputs)) |> self.device #
		# μ_sum = torch.zeros((n_edges[-1],self.n_feature_outputs),dtype=torch.float32,device=self.device)
		for i in 1:self.batch_size
			μ_sum[n_edges[i]:n_edges[i+1],:] = sum(μ[n_edges[i]:n_edges[i+1],:],dims=1)
		end
	end

	Q = self.l2_1(torch.cat((μ_sum,μ),dims=2))
	return Q
end

function Forward(self::NN,v::Matrix{Int32},w::Matrix{Int32},connectivity::Matrix{Int32},n_mu_iter=3,nm_batch=None)
	   
	# v[n_nodes,n_node_in_features]
	# w[n_edges,n_edge_in_features]
	# connectivity[n_edges,2]
	# nm_batch[BATCH_SIZE] : int

	IA,I1,I2,D = self.Connectivity(connectivity,v.shape[0])

	# if type(v) is np.ndarray
	# 	v = torch.tensor(v,dtype=torch.float32,device=self.device,requires_grad=False)
	# end

	# if type(w) is np.ndarray
	# 	w = torch.tensor(w,dtype=torch.float32,device=self.device,requires_grad=False)
	# end
	μ = zeros(Float32, (connectivity.shape[0],self.n_feature_outputs)) |> self.device
	# mu = torch.zeros((connectivity.shape[0],self.n_feature_outputs),device=self.device)

	for i in range(n_mu_iter)
		μ = getμ(self,v,μ,w,IA,I1,I2,D,mu_iter=i)
		# print("iter {0}: {1}".format(i,mu.norm(p=2)))
	end
	
	if nm_batch == None
		Q = getQ(self,μ,size(w)[1])
	else
		Q = getQ(self,μ,nm_batch)
	end
	Q = Q.flatten()

	return Q
end

function Save(self,filename,directory="")
	torch.save(self.to('cpu').state_dict(),os.path.join(directory,filename))
end

function Load(self,filename,directory="")
	self.load_state_dict(torch.load(os.path.join(directory,filename)))
end


# end of class NN


Base.@kwdef mutable struct Brain
	device::Any #CuDevice
	n_node_inputs::Int
	n_edge_inputs::Int
	batch_size::Int
	model::NN
	target_model::NN
	optimizer::Adam
	n_edge_action_type::Int
	n_step::Int
	gamma::Float64
	nstep_gamma::Float64
	temp_buffer::Deque
	capacity::Int
	buffer::Deque
	tdfunc::Function
	priority::Array{Float32,1}
	max_priority::Float32
	beta_scheduler::Function
	store_count::Int

function Brain(n_node_inputs::Int, n_edge_inputs::Int, n_feature_outputs::Int, n_action_types::Int, use_gpu::Bool)
	if use_gpu
		device = gpu
	else
		device = cpu
	end
	
	batch_size = 32
	model = NN(n_node_inputs, n_edge_inputs, n_feature_outputs, n_action_types, batch_size, use_gpu)
	target_model = deepcopy(model)
	optimizer = Adam(model.parameters, lr=1.0e-4) # RMSprop(self.model.parameters(),lr=1.0e-5)
	n_edge_action_type = n_action_types
	n_step = 3
	gamma = 1.0
	nstep_gamma = gamma^n_step
	temp_buffer = Deque{Int32}
	capacity = Int32(1E4)
	buffer = Deque{Int32} #Deque([nothing for _ in 1:capacity], capacity)
	tdfunc = Flux.mae
	priority = zeros(Float32, capacity)
	max_priority = 1.0
	beta_scheduler = progress -> 0.4 + 0.6*progress
	store_count = 0
	
	new(device, n_node_inputs, n_edge_inputs, batch_size, model, target_model, optimizer,
		n_edge_action_type, n_step, gamma, nstep_gamma, temp_buffer, capacity, buffer,
		tdfunc, priority, max_priority, beta_scheduler, store_count)
	end

end

function store_experience(
    self::Brain, c, v, w, action, reward, c_next, v_next, w_next, done, infeasible_action, stop
)
    v = Float32.(v)
    w = Float32.(w)
    v_next = Float32.(v_next)
    w_next = Float32.(w_next)
    push!(self.temp_buffer, Temp_Experience(c, v, w, action, reward))

    # Using Multistep learning
    if done || stop
        for j in 1:length(self.temp_buffer)
            nstep_return = sum(
                [self.gamma ^ (i - j) * self.temp_buffer[i].reward for i in j:length(self.temp_buffer)]
            )
            self.priority[1:end-1], self.priority[end] = self.priority[2:end], self.max_priority
            push!(
                self.buffer,
                Experience(
                    self.temp_buffer[j].c, self.temp_buffer[j].v, self.temp_buffer[j].w, 
                    self.temp_buffer[j].action, nstep_return, c_next, v_next, w_next, done, 
                    infeasible_action
                )
            )
            self.store_count += 1
        empty!(self.temp_buffer)
    elseif length(self.temp_buffer) == self.n_step
        nstep_return = sum([self.gamma ^ i * self.temp_buffer[i].reward for i in 1:length(self.temp_buffer)])
        self.priority[1:end-1], self.priority[end] = self.priority[2:end], self.max_priority
        push!(
            self.buffer,
            Experience(
                self.temp_buffer[1].c, self.temp_buffer[1].v, self.temp_buffer[1].w, self.temp_buffer[1].action, 
                nstep_return, c_next, v_next, w_next, done, infeasible_action
            )
        )
        self.store_count += 1
    end
end



function sample_batch(self, progress)
    p = self.priority ./ sum(self.priority)
    indices = rand(Categorical(p), self.batch_size, replace=false)

    weight = (p[indices] .* self.capacity) .^ (-self.beta_scheduler(progress))
    weight /= maximum(weight)

    batch = [self.buffer[i] for i in indices]

    c_batch = zeros(Int, 0, 2)
    v_batch = cat([dat.v for dat in batch]..., dims=1)
    w_batch = cat([dat.w for dat in batch]..., dims=1)
    a_batch = [dat.action for dat in batch]
    r_batch = Float32.([dat.reward for dat in batch])
    c2_batch = zeros(Int, 0, 2)
    v2_batch = cat([dat.v_next for dat in batch]..., dims=1)
    w2_batch = cat([dat.w_next for dat in batch]..., dims=1)
    done_batch = Bool.([dat.done for dat in batch])
    infeasible_a_batch = vcat([dat.infeasible_action for dat in batch]...)
    nm_batch = zeros(Int, self.batch_size+1)
    nm2_batch = zeros(Int, self.batch_size+1)

    nn = 0
    nm = 0
    for i in 1:self.batch_size
        c_batch = vcat(c_batch, batch[i].c .+ nn)
        nn += size(batch[i].v, 1)
        nm += size(batch[i].w, 1)
        nm_batch[i+1] = nm
    end
    a_batch .+= (nm_batch[1:end-1] .* self.n_edge_action_type)

    nn2 = 0
    nm2 = 0
    for i in 1:self.batch_size
        c2_batch = vcat(c2_batch, batch[i].c_next .+ nn2)
        nn2 += size(batch[i].v_next, 1)
        nm2 += size(batch[i].w_next, 1)
        nm2_batch[i+1] = nm2
    end

    return weight, indices, c_batch, v_batch, w_batch, a_batch, r_batch, c2_batch, v2_batch, w2_batch, done_batch, infeasible_a_batch, nm_batch, nm2_batch
end

function update_priority(self, indices, td_errors)
    pri = (abs.(Flux.Tracker.value(td_errors).to("cpu").data) .+ 1e-3).^0.6
    self.priority[indices] = pri
    self.max_priority = max(self.max_priority, maximum(pri))
end

function experience_replay(self, progress)
    if self.store_count < self.batch_size
        return NaN
    end

    weight, indices, c, v, w, a, r, c_next, v_next, w_next, done, infeasible_action, nm, nm_next = sample_batch(progress)
    Flux.reset!(self.optimizer)
    td_errors = calc_td_error(c, v, w, a, r, c_next, v_next, w_next, done, infeasible_action, nm, nm_next)
    loss = mean((td_errors .^ 2) .* torch.Tensor(weight) .|> device=self.device)
    Flux.back!(loss)
    step!(self.optimizer)
    update_priority(self, indices, td_errors)
    return loss.item()
end

function calc_td_error(self, c, v, w, action, r, c_next, v_next, w_next, done, infeasible_action, nm, nm_next)
    current_Q = self.model.Forward(v, w, c, nm_batch=nm)
    next_QT = self.target_model.Forward(v_next, w_next, c_next, nm_batch=nm_next) |> detach
    next_Q = self.model.Forward(v_next, w_next, c_next, nm_batch=nm_next) |> detach

    ### Without Double DQN ###
    next_QT[infeasible_action] .= -1.0e20
    Q_max_next = [maximum(next_QT[(nm_next[i]*self.n_edge_action_type)+1:nm_next[i+1]*self.n_edge_action_type]) if nm_next[i] != nm_next[i+1] else 0.0 for i in 1:self.batch_size] |> torch.tensor(device=self.device)

    # ### Using Double DQN ###
    # next_Q[infeasible_action] .= -1.0e20
    # action_next = [(nm_next[i]*self.n_edge_action_type) + argmax(next_Q[(nm_next[i]*self.n_edge_action_type)+1:nm_next[i+1]*self.n_edge_action_type]).item() for i in 1:self.batch_size]
    # Q_max_next = next_QT[action_next]

    Q_target = r + (self.nstep_gamma .* Q_max_next) .* ~done
    td_errors = tdfunc(current_Q[action], Q_target)  # In rainbow, not td_error but KL divergence loss is used to
    return td_errors
end


function decide_action(self, v, w, c, eps, infeasible_actions)
    Q = self.model.Forward(v, w, c)
    Q = detach(Q).to("cpu").numpy()

    if rand() > eps
        masked_Q = ma.masked_where(infeasible_actions, Q)
        a = argmax(masked_Q)
    else
        feasible_actions = findall(!infeasible_actions)
        a = feasible_actions[rand(1:length(feasible_actions))]
    end

    return a, Q[a]
end

mutable struct Agent
    brain::Brain
    step::Int64
    n_update::Int64
    target_update_freq::Int64
end

function Agent(n_node_inputs::Int64, n_edge_inputs::Int64, n_feature_outputs::Int64, n_action_types::Int64, use_gpu::Bool)
    brain = Brain(n_node_inputs, n_edge_inputs, n_feature_outputs, n_action_types, use_gpu)
    step = 0
    n_update = 0
    target_update_freq = TARGET_UPDATE_FREQ
    Agent(brain, step, n_update, target_update_freq)
end

function update_q_function(agent::Agent, progress::Float64)
    loss = agent.brain.experience_replay(progress)
    if agent.n_update % agent.target_update_freq == 0
        load_state_dict!(agent.brain.target_model, deepcopy(agent.brain.model.state_dict()))
    end
    agent.n_update += 1
    return loss
end

function get_action(agent::Agent, v, w, c, eps, infeasible_actions)
    action, q = agent.brain.decide_action(v, w, c, eps, infeasible_actions)
    return action, q
end

function memorize(agent::Agent, c, v, w, action, reward, c_next, v_next, w_next, ep_end, infeasible_actions, stop)
    agent.brain.store_experience(c, v, w, action, reward, c_next, v_next, w_next, ep_end, infeasible_actions, stop)
end


