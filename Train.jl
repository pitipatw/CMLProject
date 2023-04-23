
include("RL_environment.jl")

### User specified parameters ###
USE_GPU = True
N_EPISODE = 5000
#################################

## Train ##


@time env = environment.Environment(gpu=USE_GPU)
@time env.Train(N_EPISODE)
@time env.Test()
# env.agent.brain.model.Output_params()
# for i in range(5):
#     env.Test()
