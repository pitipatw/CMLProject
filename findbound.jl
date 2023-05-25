# x = rand(1000);
# y = rand(1000);
# x = collect(copy(x_train))[:]
# y = collect(copy(y_train))[:]
function find_upperbound(x::Vector{Float32}, y::Vector{Float32})
    #get the bottom left most point
    x_min, x_min_i = findmin(x)
    y_min = y[x_min_i]
    # println("x_min = $x_min, y_min = $y_min")
    #get all points to the right of the current point

    remain_points = Vector{Tuple{Float64}}()
    for i in eachindex(x)
        remain_points = vcat(remain_points, (x[i], y[i]))
    end
    # println(remain_points)
    path = [(x_min, y_min)]
    next_x = x_min
    next_y = y_min

    while length(remain_points) > 0
        nr = size(remain_points)[1]
        # println(nr)
        slope_old = -Inf
        for i in 1:nr
            # println(remain_points)
            pt = remain_points[i]
            xi , yi = copy(pt)
            # println(1)
            if xi > x_min
                slope_new = (yi - y_min)/(xi - x_min)
                # println("new slope: $slope_new and old slope: $slope_old")
                if slope_new > slope_old
                    slope_old = slope_new
                    next_x = xi
                    next_y = yi 
                    # println(3)
                else
                    # println(4)
                    continue
                end
            end
        end
        push!(path, (next_x, next_y))
        d_idx = []
        x_min = next_x
        y_min = next_y
        for i in eachindex(remain_points)
            # println(size(remain_points))
            # println(remain_points[i][1])
            if x_min >= remain_points[i][1] || 0.7*y_min >= remain_points[i][2]
                push!(d_idx, i)
            end
        end
        deleteat!(remain_points, d_idx)
        
    end
    return path
end

function find_lowerbound(x::Vector{Float32}, y::Vector{Float32})
    #get the bottom left most point
    x_min, x_min_i = findmin(x)
    y_min = y[x_min_i]
    # println("x_min = $x_min, y_min = $y_min")
    #get all points to the right of the current point

    remain_points = Vector{Tuple{Float64}}()
    for i in eachindex(x)
        remain_points = vcat(remain_points, (x[i], y[i]))
    end
    # println(remain_points)
    path = [(x_min, y_min)]
    next_x = x_min
    next_y = y_min

    while length(remain_points) > 0
        nr = size(remain_points)[1]
        # println(nr)
        slope_old = Inf
        for i in 1:nr
            # println(remain_points)
            pt = remain_points[i]
            xi , yi = copy(pt)
            # println(1)
            if xi > x_min
                slope_new = (yi - y_min)/(xi - x_min)
                # println("new slope: $slope_new and old slope: $slope_old")
                if slope_new < slope_old
                    slope_old = slope_new
                    next_x = xi
                    next_y = yi 
                    # println(3)
                else
                    # println(4)
                    continue
                end
            end
        end
        push!(path, (next_x, next_y))
        d_idx = []
        x_min = next_x
        y_min = next_y
        for i in eachindex(remain_points)
            # println(size(remain_points))
            # println(remain_points[i][1])
            if x_min >= remain_points[i][1]
                push!(d_idx, i)
            end
        end
        deleteat!(remain_points, d_idx)        
    end
    return path
end


# f0 = Figure(resolution = (1200,800))
# ax0 = Axis(f0[1,1])
# scatter!(ax0, x,y)
# f0
# path1 = find_upperbound(x,y)
# path2 = find_lowerbound(x,y)
# lines!(ax0, path1)
# lines!(ax0, path2)
# f0

"""
Nearest Neighbor Search
input as functions
"""

function nns(f; radius::Float64 = 0.001, mode::String = "upper")
    if mode == "upper"
        #get only points that above the points in the path

    end
end
