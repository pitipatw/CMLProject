

"""
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
"""


#### setup Topology Optimization for continuum #####
#get benchmark problem
# include("Benchmark1.jl")

f2 = Figure(resolution=(600, 200))
f3 = Figure(resolution=(600, 200))
# (min at 50)
# compliance_threshold = 500 # maximum compliance
# lc = [4000,3000, 2000, 1000, 900, 800, 700, 600, 500, 400, 300, 250,200, 100, 90, 80, 70, 60, 50, 49, 48, 47, 46]
lc = [350, 400, 500 , 600]
lc = 600:100:2000
for i in lc
    f2 = Figure(resolution=(600, 200))
    f3 = Figure(resolution=(600, 200))
    f2e = save_func_e[2]
    f2g = save_func_g[2]
    compliance_threshold  =  i
    println("Start loop number ", i, " with compliance_threshold = ", compliance_threshold)
E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 2.0 # downward force
rmin = 4.0 # filter radius
xmin = 0.0001 # minimum density
nx = 100
ny = 50
problem_size = (nx, ny)
ncells = prod(problem_size)
x0 = vcat(fill(1.0, ncells), fill(80.0, ncells)) # initial design
# println(size(x0))
p = 1.0 # penalty

# Young's modulus interpolation for compliance
# penalty1 = TopOpt.PowerPenalty(1.0) # take young modulus in each material 
# interp1 = MaterialInterpolation(Es, penalty1)

# # density interpolation for mass constraint
# penalty2 = TopOpt.PowerPenalty(1.0) #no penalty.
# interp2 = MaterialInterpolation(densities, penalty2)




# problem = PointLoadCantilever(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)
problem = HalfMBB(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)


solver = FEASolver(Direct, problem; xmin=xmin)

cheqfilter = DensityFilter(solver; rmin=rmin)
stress = TopOpt.von_mises_stress_function(solver)
comp = TopOpt.Compliance(solver)

function obj(x)
    # function constr(x)
    fc = x[Int32(length(x) / 2)+1:end]
    den = x[1:Int32(length(x) / 2)]
    gwp = [x[1] for x in f2g.(fc)]
    # minimize volume
    return sum(cheqfilter(PseudoDensities(den .* gwp))) / length(x)*2 #- 0.4
end
function constr(x)
    # function obj(x)
    # compliance upper-bound
    f = x[Int32(length(x) / 2)+1:end]
    E = f2e(f)
    den = x[1:Int32(length(x) / 2)]
    return comp(cheqfilter(PseudoDensities(den .* E))) - compliance_threshold
end


@show constr(x0)
obj(x0)
gradient(constr, x0)
gradient(obj, x0)
m = TopOpt.Model(obj)
addvar!(m, vcat(zeros(length(x0) ÷ 2), 10 * ones(length(x0) ÷ 2)), vcat(ones(length(x0) ÷ 2), 100 * ones(length(x0) ÷ 2)))
Nonconvex.add_ineq_constraint!(m, constr)

options = MMAOptions(; maxiter=1000, tol=Tolerance(; kkt=1e-4, x=1e-4, f=1e-4))
TopOpt.setpenalty!(solver, p)
@time r = Nonconvex.optimize(
    m, MMA87(; dualoptimizer=ConjugateGradient()), x0; options=options
)


Amin = r.minimizer[1:Int32(length(r.minimizer) / 2)]
fmin = r.minimizer[Int32(length(r.minimizer) / 2)+1:end]
fmax = maximum(fmin)
fmin_n = fmin ./ fmax
@show obj(r.minimizer)
@show constr(r.minimizer)

@show maximum(stress(cheqfilter(PseudoDensities(Amin))))
Amin = cheqfilter(PseudoDensities(Amin)).x
fmin = cheqfilter(PseudoDensities(fmin)).x
# fig1 = visualize(problem; solver.u, topology=Amin, default_exagg_scale=0.0, scale_range=10.0)
# Makie.display(fig1)

# fig2 = visualize(problem; solver.u, topology=fmin_n, default_exagg_scale=0.0, scale_range=10.0)
# Makie.display(fig2)

Am= fmin_n
mapping = Array{Int64,2}(undef, ncells, 2)
for i in 1:nx*ny
    x = mod(i, nx)
    y = div(i, nx)+1
    if x == 0 
        x = nx
        y = y-1
    end
    mapping[i, :] = [x,y]
end

# ax3 = Axis(f2[1, 1])
# scatter!(ax3, mapping[:, 2], mapping[:, 1], color=y)


ax4, hm1 = heatmap(f2[1, 1], mapping[:, 1], mapping[:, 2], Amin)



ax5, hm2 = heatmap(f3[1, 1], mapping[:, 1], mapping[:, 2], fmin)
# ax6, hm3 = heatmap(f2[3, 1], mapping[:, 1], mapping[:, 2], fmin.*Amin)

ax4.title = "Area (desity)"
ax5.title = "fc′"
ax4.aspect = 3
ax5.aspect = 3
# cbar1 = Colorbar(f2[1, 2])
cbar1 = Colorbar(f2[1,2], hm1)
cbar2 = Colorbar(f3[1,2], hm2)

# cbar3 = Colorbar(f2[3,2], hm3)
f2
f3
save("fil_area"*string(compliance_threshold)*".png", f2)
save("fil_fc"*string(compliance_threshold)*".png", f3)

end

f2
f3
# # end
