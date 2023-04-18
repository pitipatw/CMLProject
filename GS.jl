#with node_points
# #node number -> [x y z]
using LinearAlgebra
# using Makie, GLMakie

function getGS(lx::Float64,ly::Float64,lz::Float64 ,nx::Int64,ny::Int64,nz::Int64 ; origin::Vector{Float64} = [0.0,0.0,0.0])
    node_points = Dict{Int64,Vector{Float64}}()
    global node_counter = 0 
    for i in 0:(nz-1)
        for j in 0:(ny-1)
            for k in 0:(nx-1)
                global node_counter += 1
                node_points[node_counter] = [origin[1]+ k*lx, origin[2]+j*ly, origin[3]+i*lz]
            end
        end
    end
    GS = Dict{Int64,Vector{Int64}}()
    global element_counter = 0 
    for i in eachindex(node_points)
        current_pt = node_points[i]
        #list of unit vectors 
        lov = Dict{Vector{Float64},Vector{Int64}}()
        lod = Dict{Vector{Float64},Float64}()
        for j in eachindex(node_points)
            if j>i
                next_pt = node_points[j]
                #unit vector from current_pt to next_pt
                uv = (next_pt - current_pt)/norm(next_pt - current_pt)
                dis = norm(next_pt - current_pt)
                # println("---")
                # println("i ",i)
                # println("j ",j)
                #check if uv is in lov
                check_uv = false
                for k in keys(lov)
                    if uv ≈ k
                        check_uv = true
                        if dis < lod[uv]
                            lod[uv] = dis
                            lov[uv] = [i,j]
                            # println("edit")
                        end
                    end
                end
                if !check_uv   
                    lov[uv] = [i,j]
                    lod[uv] = dis
                    # println("add")
                end
            end
        end
        # println(lov)
        for k in eachindex(lov)
        global element_counter += 1 
            # println(typeof(lov[i]))
            # println(element_counter)
            GS[element_counter] = lov[k]
        end

    end
    return GS
end

function getGS(node_points::Dict{Int64,Vector{Float64}})
    GS = Dict{Int64,Vector{Int64}}()
    global element_counter = 0 
    for i in eachindex(node_points)
        current_pt = node_points[i]
        #list of unit vectors 
        lov = Dict{Vector{Float64},Vector{Int64}}()
        lod = Dict{Vector{Float64},Float64}()
        for j in eachindex(node_points)
            if j>i
                next_pt = node_points[j]
                #unit vector from current_pt to next_pt
                uv = (next_pt - current_pt)/norm(next_pt - current_pt)
                dis = norm(next_pt - current_pt)
                # println("---")
                # println("i ",i)
                # println("j ",j)
                #check if uv is in lov
                check_uv = false
                for k in keys(lov)
                    if uv ≈ k
                        check_uv = true
                        if dis < lod[k]
                            lod[k] = dis
                            lov[k] = [i,j]
                            # println("edit")
                        end
                    end
                end
                if !check_uv   
                    lov[uv] = [i,j]
                    lod[uv] = dis
                    # println("add")
                end
            end
        end
        # println(lov)
        for k in eachindex(lov)
        global element_counter += 1 
            # println(typeof(lov[i]))
            # println(element_counter)
            GS[element_counter] = lov[k]
        end

    end
    return GS
end

# #Element visualization
# F1 = Figure(resolution = (1000, 1000))
# ax = Axis3(F1[1,1])
# countdown = 16
# for i in eachindex(GS)
#     # countdown -= 1

#     x1 = node_points[GS[i][1]][1]
#     y1 = node_points[GS[i][1]][2]
#     z1 = node_points[GS[i][1]][3]
#     x2 = node_points[GS[i][2]][1]
#     y2 = node_points[GS[i][2]][2]
#     z2 = node_points[GS[i][2]][3]

#     # ax = Axis3(F1[div(countdown,4), rem(countdown,4)])
#     Makie.lines!( ax, [x1,x2], [y1,y2], [z1,z2], linewidth = 10 , 
#     color = (:red, 0.1))
#     xlims!(0, 5)
#     ylims!(0, 5)
#     zlims!(0, 5)
#     # limits = (0, 5, 0, 5, 0, 5))
#     # if countdown <= 0 
#     #     break
#     # end
# end

# display(F1)