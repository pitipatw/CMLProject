

"""
Let's do Area optimizaiton first, get a very nice area, then do the fc optimization
"""


#### setup Topology Optimization for continuum #####
#get benchmark problem
# include("Benchmark1.jl")




# (min at 50)
compliance_threshold = 800 # maximum compliance
# lc = [4000, 2000, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 49, 48, 47, 46]
# for i in lc
#     compliance_threshold  =  i
begin
E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 4.0 # filter radius
xmin = 0.0001 # minimum density
problem_size = (160, 40)
x0 = vcat(fill(1.0, prod(problem_size))) # initial design
println(size(x0))
p = 4.0 # penalty

# problem = PointLoadCantilever(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)
problem = HalfMBB(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)


solver = FEASolver(Direct, problem; xmin=xmin)

cheqfilter = DensityFilter(solver; rmin=rmin)
stress = TopOpt.von_mises_stress_function(solver)
comp = TopOpt.Compliance(solver)

function obj(x)
    # function constr(x)
    # fc = x[Int32(length(x) / 2)+1:end]
    # den = x[1:Int32(length(x) / 2)]
    # gwp = [x[1] for x in f2g.(fc)]
    # minimize volume
    return sum(cheqfilter(PseudoDensities(x))) / length(x) #- 0.4
end
function constr(x)
    # function obj(x)
    # compliance upper-bound
    # f = x[Int32(length(x) / 2)+1:end]
    # den = x[1:Int32(length(x) / 2)]
    return comp(cheqfilter(PseudoDensities(x))) - compliance_threshold
end


@show constr(x0)
obj(x0)
gradient(constr, x0)
gradient(obj, x0)
m = TopOpt.Model(obj)
addvar!(m, zeros(length(x0)), ones(length(x0)))
Nonconvex.add_ineq_constraint!(m, constr)

options = MMAOptions(; maxiter=1000, tol=Tolerance(; kkt=1e-4, x=1e-4, f=1e-4))
TopOpt.setpenalty!(solver, p)
@time r = Nonconvex.optimize(
    m, MMA87(; dualoptimizer=ConjugateGradient()), x0; options=options
)




Amin = r.minimizer
@show obj(r.minimizer)
@show constr(r.minimizer)

@show maximum(stress(cheqfilter(PseudoDensities(Amin))))
topology = cheqfilter(PseudoDensities(Amin)).x

A0 = topology #will be fixed
ncells = length(A0)
mapping = Array{Int64,2}(undef, ncells, 2)
for i in 1:160*40
    x = mod(i, 160)
    y = div(i, 160)+1
    if x == 0 
        x = 160
        y = y-1
    end
    mapping[i, :] = [x,y]
end

end


f0 = Figure(resolution=(600, 200))
ax4, hm1 = heatmap(f0[1, 1], mapping[:, 1], mapping[:, 2], A0)
ax4.title = "Area (desity)"
ax4.aspect = 3
cbar1 = Colorbar(f0[1,2], hm1)
f0


f0

save("first_step-final.png", f0)




x0 = vcat(fill(100.0, prod(problem_size))) # initial design
println(size(x0))
p = 1.0 # penalty

rmin = 1.0
compliance_threshold = 350.0
compliance_threshold = 800.0

problem = HalfMBB(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)

solver = FEASolver(Direct, problem; xmin=xmin)

cheqfilter = DensityFilter(solver; rmin=rmin)
stress = TopOpt.von_mises_stress_function(solver)
comp = TopOpt.Compliance(solver)

function obj(x)
    # function constr(x)
    # fc = x[Int32(length(x) / 2)+1:end]
    # den = x[1:Int32(length(x) / 2)]
    gwp = f2g.(x)
    # minimize volume
    return sum(cheqfilter(PseudoDensities(gwp.*A0))) / length(x) #- 0.4
    # return sum(PseudoDensities(gwp.*A0)) / length(x) #- 0.4
end

function constr(x)
    # function obj(x)
    # compliance upper-bound
    # f = x[Int32(length(x) / 2)+1:end]
    # den = x[1:Int32(length(x) / 2)]
    E = f2e(x)
    return comp(cheqfilter(PseudoDensities(E.*A0))) - compliance_threshold
    # return comp(PseudoDensities(E.*A0)) - compliance_threshold
end


@show constr(x0)
obj(x0)
gradient(constr, x0)
gradient(obj, x0)
m = TopOpt.Model(obj)
addvar!(m, 1*ones(length(x0)), 100 .*ones(length(x0)))
Nonconvex.add_ineq_constraint!(m, constr)

options = MMAOptions(; maxiter=1000, tol=Tolerance(; kkt=1e-4, x=1e-4, f=1e-4))
TopOpt.setpenalty!(solver, p)
@time r = Nonconvex.optimize(
    m, MMA87(; dualoptimizer=ConjugateGradient()), x0; options=options
)




fmin = r.minimizer
# fmin = r.minimizer[Int32(length(r.minimizer) / 2)+1:end]
# fmax = maximum(fmin)
# fmin_n = fmin ./ fmax
@show obj(r.minimizer)
@show constr(r.minimizer)

@show maximum(stress(cheqfilter(PseudoDensities(Amin))))
topology_A = cheqfilter(PseudoDensities(Amin)).x
topology_f = cheqfilter(PseudoDensities(fmin)).x





# fig1 = visualize(problem; solver.u, topology=Amin, default_exagg_scale=0.0, scale_range=10.0)
# Makie.display(fig1)

# fig2 = visualize(problem; solver.u, topology=fmin_n, default_exagg_scale=0.0, scale_range=10.0)
# Makie.display(fig2)

mapping = Array{Int64,2}(undef, ncells, 2)
for i in 1:160*40
    x = mod(i, 160)
    y = div(i, 160)+1
    if x == 0 
        x = 160
        y = y-1
    end
    mapping[i, :] = [x,y]
end

# ax3 = Axis(f2[1, 1])
# scatter!(ax3, mapping[:, 2], mapping[:, 1], color=y)



f2 = Figure(resolution = (600 , 200)) 
ax2, hm2 = heatmap(f2[1, 1], mapping[:, 1], mapping[:, 2], Amin)
ax2.title = "Area (desity)"
ax2.aspect = 3
cbar2 = Colorbar(f2[1,2], hm2)
f2



f3 = Figure(resolution=(600, 200))
ax3, hm1 = heatmap(f3[1, 1], mapping[:, 1], mapping[:, 2], r.minimizer.*Amin)
ax3.title = "fc′"
ax3.aspect = 3
cbar1 = Colorbar(f3[1,2], hm1)
f3

begin
    println("#"^50)
    println("Part4 report:")
    println("Part4 has compliance threshold: ", compliance_threshold)
    println("Part4 has compliance: ", constr(r.minimizer) + compliance_threshold)
    println("Part4 has embodied carbon: ", sum(Amin .* f2g.(fmin)))
    println("Part4 has penalty: ", p)
    println("Part4 has objective: ", obj(r.minimizer))
    println("END of Part2")
    println("#"^50)
end

save("muti_fc"*string(compliance_threshold)*".png", f3)


#quick histogram
f_hist = Figure(resolution=(600, 200))
ax_hist = Axis(f_hist[1, 1])
hist(topology_f, bins=100)


# end

# # end