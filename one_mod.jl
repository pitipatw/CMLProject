using TopOpt, Test, Zygote, Test
using Makie, GLMakie
using Colors, ColorSchemes

"""
Ultimate plan is to make the decesion variable just 1 per cell -> directly means modulus of the concrete.

plot another heatmap, but with the value as the max decision value, so we know if ut's hesitaing or not
"""


#Dummy Function definition
m = Chain(Dense(1 => 50, sigmoid), Dense(50 => 1))
begin
    fc′ = 30:5:90
    f2e = x -> 4700 * sqrt(x)
    f2g = x -> 0.5 * sqrt(x) / 10 
    
    f2e = x -> 4700 * x
    f2g = x -> 0.5 * x / 10

    f2g = sgt
    #function plot
    Es = vcat([1e-5], f2e.(fc′)) / 10000
    densities = vcat([0.0], f2g.(fc′))

    f1 = Figure(resolution=(1800, 600))
    ax1 = Axis(f1[1, 1])
    ax1.yticklabelsize = 30
    ax1.xticklabelsize = 30
    ax1.ylabel = "Young's modulus (MPa)"
    ax1.xlabel = "Concrete strength (MPa)"
    ax1.title = "Young's modulus vs concrete strength"
    ax1.titlesize = 40
    ax1.ylabelsize = 30
    ax1.xlabelsize = 30
    lines!(ax1, fc′, Es[2:end], color=:red, label="E")

    ax2 = Axis(f1[1, 2])
    lines!(ax2, fc′, f2g.(fc′), color=:blue, label="density")
    ax2.yticklabelsize = 30
    ax2.xticklabelsize = 30
    ax2.ylabel = "GWP"
    ax2.xlabel = "Concrete strength (MPa)"
    ax2.title = "GWP vs concrete strength"
    ax2.titlesize = 40
    ax2.ylabelsize = 30
    ax2.xlabelsize = 30
    f1
end



Es = [1e-5, 1.0, 3.0] # Young's moduli of 3 materials (incl. void)
densities = [0.0, 0.5, 1.0]


Es = vcat([1e-5], f2e.(fc′)) / 10000
densities = vcat([0.0], f2g.(fc′))

f2e = m 
f2g = m
f2g(10)
begin
    nmats = length(Es)
    nu = 0.3 # Poisson's ratio
    f = 5.0 # downward force
    # problem definition
    problem = PointLoadCantilever(
        Val{:Linear}, # order of bases functions
        (160, 40), # number of cells
        (1.0, 1.0), # cell dimensions
        1.0, # base Young's modulus
        nu, # Poisson's ratio
        f, # load
    )
    ncells = TopOpt.getncells(problem)

    # FEA solver
    solver = FEASolver(Direct, problem; xmin=0.001)

    # density filter definition
    filter = DensityFilter(solver; rmin=3.0)

    # compliance function
    comp = Compliance(solver)

    # Young's modulus interpolation for compliance
    penalty1 = TopOpt.PowerPenalty(3.0) # takes young modulus in each material 
    interp1 = MaterialInterpolation(Es, penalty1)

    # density interpolation for mass constraint
    penalty2 = TopOpt.PowerPenalty(1.0) #no penalty.
    interp2 = MaterialInterpolation(densities, penalty2)

    # objective function -> weight minimization
    obj = y -> begin
        # _rhos = interp2(MultiMaterialVariables(y, nmats)) #rho is the density
        yy = [[y1] for y1 in y]
        ρ = f2g.(yy)
        ρ = [y[1] for y in ρ]
        # return sum(_rhos.x) / ncells # elements have unit volumes, 0.4 is the target.
        return sum(ρ)/ncells
    end
    # PseudoDensities(y)

    # initial decision variables as a vector
    # y0 = zeros(ncells * (nmats - 1))
    y0 = 40*ones(ncells)
    # testing the objective function
    @show obj(y0)
    # testing the gradient
    Zygote.gradient(obj, y0)

    # compliance constraint

    # list_of_stress_lim = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    list_of_stress_lim = [0.25]
    
    comp_lim = list_of_stress_lim[1]
    comp_lim = 0.035
    #compliance constraint
    constr = y -> begin
        println(y)
        println(typeof(y))
    
        # x = tounit(MultiMaterialVariables(y, nmats))
        y = [[y1] for y1 in y]
        # _E = interp1(filter(x))
        y = f2e.(y)
        y = [y[1] for y in y]
        println(y)
        println(typeof(y))
        _E = PseudoDensities(abs.(y))
        println(comp(_E))
        # println("E: ", _E)
        return comp(_E) - comp_lim #take that and multiply by the volume
    end

    # testing the mass constraint
    @test constr(y0) < 1e-6
    # testing the gradient
    Zygote.gradient(constr, y0);

end ;


begin
    # building the optimization problem
    model = Model(obj)
    # addvar!(model, fill(-10.0, length(y0)), fill(10.0, length(y0)))
    addvar!(model, fill(1, length(y0)), fill(90, length(y0)))
    add_ineq_constraint!(model, constr)

    # optimization settings
    alg = MMA87()
    options = MMAOptions(; s_init=0.1, tol=Tolerance(; kkt=1e-3))
end;

begin
    # solving the optimization problem
    @time res = optimize(model, alg, y0; options)
    y = res.minimizer

    # testing the solution
    @test constr(y) < 1e-6

    # x = TopOpt.tounit(reshape(y, ncells, nmats - 1))  #reshape into a matrix with rows of each elements, and columns of each material propabilities/
    # sum(x[:, 2:3]) / size(x, 1) # the non-void elements as a ratio
    # @test all(x -> isapprox(x, 1), sum(x; dims=2))

end;
y

begin

    # dec_val = Vector{Float64}(undef, ncells) #decision values
    # dec_mat = Vector{Int64}(undef, ncells) # contains the decision material of each cell
    # for i in eachindex(dec_val)
    #     dec_val[i], dec_mat[i] = findmax(x[i, :])
    # end

    #create mapping from 1- 6400 to grid 1600 * 40 
    mapping = Array{Int64,2}(undef, ncells, 2)
    for i in 1:160*40
        mapping[i, :] = [div(i, 160) + 1, mod(i, 160)]
    end

    f2 = Figure(resolution=(1000, 3000))
    ax3 = Axis(f2[1, 1])
    scatter!(ax3, mapping[:, 2], mapping[:, 1], color=y)


    ax4, hm1 = heatmap(f2[2,1],mapping[:, 2], mapping[:, 1], y)
    ax5, hm2 = heatmap(f2[3,1],mapping[:, 2], mapping[:, 1], y)
    
    cbar1 = Colorbar(f2[1, 2])
    cbar2 = Colorbar(f2[2, 2], hm1)
    cbar3 = Colorbar(f2[3, 2], hm2)
    f2
end



ax3.title = "penalty 3"
save("penalty3.png", f2)
    


cbar.ticks = ([-0.66, 0, 0.66], ["negative", "neutral", "positive"])
    ax2.title = "comp: " * string(comp_lim)
    ax2.title = "3mat"
    using Colors, ColorSchemes
    figure = (; resolution=(600, 400), font="CMU Serif")
    axis = (; xlabel=L"x", ylabel=L"y", aspect=DataAspect())
    #cmap = ColorScheme(range(colorant"red", colorant"green", length=3))
    # this is another way to obtain a colormap, not used here, but try it.
    mycmap = ColorScheme([RGB{Float64}(i, 1.5i, 2i) for i in [0.0, 0.25, 0.35, 0.5]])
    fig, ax2, pltobj = heatmap(rand(-1:1, 20, 20);
        colormap=cgrad(mycmap, 3, categorical=true, rev=true), # cgrad and Symbol, mycmap
        axis=axis, figure=figure)
    cbar = Colorbar(fig[1, 2], pltobj, label="Categories")
    cbar.ticks = ([-0.66, 0, 0.66], ["negative", "neutral", "positive"])
    f1
    f2
    optobj = obj(y)
    text!("$optobj")
    strval = split(string(comp_lim), ".")
    # name = "aaafc_adjusted_multimat_compConts"*strval[1]*strval[2]*".png"
    save(name, f2)


f1
f2