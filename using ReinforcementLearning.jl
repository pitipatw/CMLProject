using ReinforcementLearning

Base.@kwdef mutable struct TopOptEnv <: AbstractEnv
    reward ::Union{Float64,Nothing} = nothing
end

Main.TopOptEnv

#these are defining functions for the environment
RLBase.action_space(env::TopOptEnv) = (:PowerRich, :MegaHaul, :Hi)

RLBase.action_space(env::TopOptEnv) = (1,2,3)

RLBase.reward(env::TopOptEnv) = env.reward

RLBase.state(env::TopOptEnv) = !isnothing(env.reward)

RLBase.state_space(env::TopOptEnv) = [false, true]

RLBase.is_terminated(env::TopOptEnv) = !isnothing(env.reward)  #should be -1

RLBase.reset!(env::TopOptEnv) = env.reward = nothing

function (x::TopOptEnv)(action)
    #take the action, calculate the reward
    #In this case, remove element i from the truss structure.
    #x.reward = reward(x, action)
    if action == :PowerRich
        x.reward = rand() < 0.01 ? 100_000_000 : -10
    elseif action == :MegaHaul
        x.reward = rand() < 0.05 ? 1_000_000 : -10
    elseif isnothing(action) x.reward = 0
    elseif action ==:Hi
        x.reward = rand() < 0.1 ? 100_000 : -10
    else
        x.reward = 1000000. #@error "unknown action of $action"
    end
end

env = TopOptEnv()

RLBase.test_runnable!(env)

a = EpsilonGreedyExplorer(0.1)
run(RandomPolicy(action_space(env)), env, StopAfterEpisode(1_000))
hook = TotalRewardPerEpisode()

run(RandomPolicy(action_space(env)), env, StopAfterEpisode(1_000), hook)


a = EpsilonGreedyExplorer(1)
a([1 ,2, 3 ,50 , 800 , 1,80,120])  

policy1 = QBasedPolicy(
           learner = MonteCarloLearner(;
                   approximator=TabularQApproximator(
                       ;n_state = 3,
                       n_action = 3,
                       opt = InvDecay(1.0)
                   )
               ),
           explorer = EpsilonGreedyExplorer(0.1)
       )

hook = TotalRewardPerEpisode()
run(policy1(action_space(env)) , env, StopAfterEpisode(1_000), hook)