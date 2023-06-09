# for Topology optimization
using TopOpt, LinearAlgebra, StatsFuns

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
# include("Benchmark2.jl")

f2e = x -> sqrt.(x) ./ 10
f2g_MX
f2g_IN
predict_pes_sig
predict_opt_sig
# f2g = predict_pes_sig
f2g = predict_opt_sig
# f2g = f2g_MX
# f2g = x-> (x./100).^4


xval = 10:1:100
f_check = Figure(resolution=(800, 800))
ax_check1 = Axis(f_check[1, 1], xlabel="x", ylabel="y", title="check")
ax_check2 = Axis(f_check[1, 2], xlabel="x", ylabel="y", title="check")
lines!(ax_check1, xval, f2g.(xval), color=:red)
lines!(ax_check2, xval, f2e.(xval), color=:blue)
f_check

maximum(f2g.(xval))

# (min at 50)
# compliance_threshold = 500 # maximum compliance
# lc = [4000,3000, 2000, 1000, 900, 800, 700, 600, 500, 400, 300, 250,200, 100, 90, 80, 70, 60, 50, 49, 48, 47, 46]
lc = 800
# for i in lc
f2 = Figure(resolution=(600, 200))
f3 = Figure(resolution=(600, 200))

# f2g = save_func_g[2]
compliance_threshold = lc
# println("Start loop number ", i, " with compliance_threshold = ", compliance_threshold)
E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 4.0 # filter radius
xmin = 0.0001 # minimum density
nx = 160
ny = 40
problem_size = (nx, ny)
ncells = prod(problem_size)
x0 = vcat(fill(1.0, ncells), fill(100.0, ncells)) # initial design
# println(size(x0))
p = 4.0 # penalty

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
    gwp = f2g.(fc)
    # minimize volume
    return sum(cheqfilter(PseudoDensities(den)) .* gwp) / length(den)
end
function constr(x)
    # function obj(x)
    # compliance upper-bound
    f = x[Int32(length(x) / 2)+1:end]
    E = f2e(f)
    den = x[1:Int32(length(x) / 2)]
    DUM = PseudoDensities(cheqfilter(PseudoDensities(den.*E)))
    return comp(DUM) - compliance_threshold
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
# Amin = cheqfilter(PseudoDensities(Amin)).x
# fmin = cheqfilter(PseudoDensities(fmin)).x
# fig1 = visualize(problem; solver.u, topology=Amin, default_exagg_scale=0.0, scale_range=10.0)
# Makie.display(fig1)

# fig2 = visualize(problem; solver.u, topology=fmin_n, default_exagg_scale=0.0, scale_range=10.0)
# Makie.display(fig2)

mapping = Array{Int64,2}(undef, ncells, 2)
for i in 1:nx*ny
    x = mod(i, nx)
    y = div(i, nx) + 1
    if x == 0
        x = nx
        y = y - 1
    end
    mapping[i, :] = [x, y]
end

# ax3 = Axis(f2[1, 1])
# scatter!(ax3, mapping[:, 2], mapping[:, 1], color=y)

f2 = Figure(resolution=(600, 200))
f3 = Figure(resolution=(600, 200))
ax4, hm1 = heatmap(f2[1, 1], mapping[:, 1], mapping[:, 2], Amin)



ax5, hm2 = heatmap(f3[1, 1], mapping[:, 1], mapping[:, 2], fmin)
# ax6, hm3 = heatmap(f2[3, 1], mapping[:, 1], mapping[:, 2], fmin.*Amin)

ax4.title = "Area (desity)"
ax5.title = "fc′"
ax4.aspect = 3
ax5.aspect = 3
# cbar1 = Colorbar(f2[1, 2])
cbar1 = Colorbar(f2[1, 2], hm1)
cbar2 = Colorbar(f3[1, 2], hm2)

# cbar3 = Colorbar(f2[3,2], hm3)
f2
f3
save("Part2_A" * string(compliance_threshold) * ".png", f2)
save("Part2_FC" * string(compliance_threshold) * ".png", f3)

# end

f2

f3
begin
    println("#"^50)
    println("Part2 report:")
    println("Part2 has compliance threshold: ", compliance_threshold)
    println("Part2 has compliance: ", constr(r.minimizer) + compliance_threshold)
    println("Part2 has embodied carbon: ", sum(Amin .* f2g.(fmin)))
    println("Part2 has penalty: ", p)
    println("Part2 has objective: ", obj(r.minimizer))
    println("END of Part2")
    println("#"^50)
end
# # end
