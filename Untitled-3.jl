
s_lin = EpsilonGreedyExplorer(kind=:linear, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100)
plot([RLCore.get_ϵ(s_lin, i) for i in 1:500], label="linear epsilon")
s_exp = EpsilonGreedyExplorer(kind=:exp, ϵ_init=0.9, ϵ_stable=0.1, warmup_steps=100, decay_steps=100)
plot!([RLCore.get_ϵ(s_exp, i) for i in 1:500], label="exp epsilon")

for i in 1:500
    println(RLCore.get_ϵ(s_lin, i))
end