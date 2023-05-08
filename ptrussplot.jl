using Makie, GLMakie

function ptrussplot(node_points::Dict{Int64, Any} , elements::Dict{Int64, Tuple{Int64, Int64}}, x0 ::Vector{Float64})
    ncells = size(x0,1)/2
    A = x0[1:ncells]
    E = x0[ncells+1:end]
    f1 = Figure(resolution = (800, 600))
    ax1 = Axis(f1[1, 1])
    for (i, (n1, n2)) in elements
        x1, y1 = node_points[n1]
        x2, y2 = node_points[n2]
        lines!(ax1, [x1, x2], [y1, y2], color = E[i], linewidth = A[i])
    end

end