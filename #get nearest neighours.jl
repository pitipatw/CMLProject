#get nearest neighours

# f is a function 
#this is for opt, so pick the uppep
dis = -0.02
points = train_data
f = predict_opt_sig 
f = predict_pes_sig
data = Vector{Vector{Float64}}()
for i =1:size(points)[1]
    x = points[i,1]
    y = points[i,2]
    if (y- f(x))> dis
        push!(data, [x,y])
    end
end

#plot the original data
f1 = Figure(resolution = (600, 600))
ax1 = Axis(f1[1,1])
scatter!(ax1, points[:,1], points[:,2], label = "original data" , color = :blue)

#plot the data that is close to the function
scatter!(ax1, [x for (x,y) in data], [y for (x,y) in data], label = "data close to function", color = :red)
lines!(ax1 , 12:0.2:80, f.(12:0.2:80), label = "function", color = :green)
lines!(ax1, 12:0.2:80, f.(12:0.2:80) .+ dis, label = "function + dis", color = :black)
f1

