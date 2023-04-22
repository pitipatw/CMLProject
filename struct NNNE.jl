struct NNNE
    x::Int32
    y::Int32
end


@time for m in fieldnames(typeof(abc))
    println(m)
    println(getfield(abc, m))
end



d = Flux.Dense(10,10,bias = false)
rand(Normal(0,0.05))

n_edge_inputs  = 2
n_feature_outputs = 3

function edge_model(n_edge_inputs, n_feature_outputs)
    return Chain(
        Dense(n_edge_inputs, 10;init = Flux.glorot_normal),
        Dense(10, 10, relu),
        Dense(10, n_feature_outputs)
    )
end

C = edge_model(n_edge_inputs, n_feature_outputs)
print(C[1].weight)

Base.@kwdef struct test2
    a :: Int32
    b :: Int32
    c = a + 10*b
end

