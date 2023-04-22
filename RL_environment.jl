using Random
Random.seed!(1000)
using deepcopy
import Plotter
import Agent

### User specified parameters ###
include("truss_env.jl")
const N_EDGE_FEATURE = 100
const RECORD_INTERVAL = 10
#################################

struct Environment
    env::Truss
    n_edge_action::Int64
    n_whole_action::Int64
    agent::Agent
end

function Environment(gpu::Bool)
    env = Truss()
    _,v,w,_ = reset(env)
    agent = Agent(v_size=size(v,2),w_size=size(w,2),n_edge_feature=N_EDGE_FEATURE,n_edge_action=1,gpu=gpu)
    if gpu
        agent.brain.model = move(agent.brain.model, "cuda")
    end
    return Environment(env,1,0,agent)
end

function Train(env::MyEnvironment, agent::MyAgent, n_episode::Int)

    RECORD_INTERVAL = 100

    history = zeros((3, n_episode ÷ RECORD_INTERVAL))
    best_score = Inf
    best_scored_iteration = -1
    best_model = deepcopy(agent.brain.model)
    n_analysis_until_best = 0
    n_analysis = 0
    eps0 = 0.2

    for episode in 1:n_episode

        c, v, w, infeasible_a = reset(env)
        total_reward = 0.0
        aveQ = 0.0
        aveloss = 0.0

        for t in 1:env.nm
            action, q = get_action(agent, v, w, c, eps0*(n_episode-episode)/n_episode, infeasible_a)
            aveQ += q
            c_next, v_next, w_next, reward, ep_end, infeasible_a, _ = step(env, action)
            memorize(agent, c, v, w, action, reward, c_next, v_next, w_next, ep_end, infeasible_a, t==env.nm)
            c = copy(c_next)
            v = copy(v_next)
            w = copy(w_next)
            aveloss += update_q_function(agent, episode/n_episode)
            total_reward += reward
            if ep_end
                break
            end
        print("episode $(lpad(episode,4)): step=$(lpad(t+1,3)) reward=$(lpad(total_reward,+5,1)) aveQ=$(lpad(aveQ/(t+1),+7,2)) loss=$(lpad(aveloss/(t+1),+7,2))")

        n_analysis += (env.steps + 1)

        if mod(episode, RECORD_INTERVAL) == RECORD_INTERVAL - 1
            score = 1.0
            for i in [1, 2]
                c, v, w, infeasible_a = reset(env, test=i)
                total_reward = 0.0
                for t in 1:env.nm
                    action, _ = get_action(agent, v, w, c, 0.0, infeasible_a)
                    c, v, w, reward, ep_end, infeasible_a, _ = step(env, action)
                    total_reward += reward
                    if ep_end
                        break
                        end
                    end
                score *= total_reward  # !!! Be careful to the sign of value
                history[i, episode ÷ RECORD_INTERVAL] = total_reward
            if score <= best_score
                best_score = score
                best_scored_iteration = episode
                best_model = deepcopy(agent.brain.model)
                n_analysis_until_best = n_analysis
                end
            history[1, episode ÷ RECORD_INTERVAL] = score
            end
        end

    with open("result/info.txt", "w") do f
        write(f, "total number of analysis: $(n_analysis) \n")
        write(f, "total number of analysis until best: $(n_analysis_until_best) \n")
        write(f, "top-scored iteration: $(best_scored_iteration + 1) \n")
        end

    graph(history)

    writedlm("result/score.csv", history, ',')
    save(best_model, "trained_model_$(string(env))")

    end

    function Test(self)
        c, v, w, infeasible_a = self.env.reset(test=2)
        self.agent = agent.Agent(size(v)[1], size(w)[1], N_EDGE_FEATURE, self.n_edge_action, false)
        self.agent.brain.model.Load(filename="trained_model_$(env.__name__)")
    
        total_reward = 0.0
        for i = 1:self.env.nm
            action, _ = self.agent.get_action(v, w, c, 0.0, infeasible_a)
            c, v, w, reward, ep_end, infeasible_a, _ = self.env.step(action)
            total_reward += reward
            if ep_end
                break
            end
        end
    
        println("total rewards: $(total_reward)")
    end

    