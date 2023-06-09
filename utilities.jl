using DataFrames


function removeNothing(df::DataFrame)
    removed_columns = Vector{String}()
    for i in names(df)
        remove_chk = false
        # println(i)
        #check if there is a vector in the column
        #criterias to remove a column

        if typeof(df[!, i]) == Vector{Vector{Any}}
            println("Column $i is a vector of vector{Any}")
            remove_chk = true
        elseif (df[!, i] .!= nothing) == 0
            println("Column $i is nothing")
            remove_chk = true

        elseif typeof(df[!, i]) == Vector{Any}
            println("Column $i is a vector of vector{Any}")
            remove_chk = true

        elseif typeof(df[!, i]) == Vector{Nothing}
            println("Column $i is a vector of vector{Any}")
            remove_chk = true


        elseif typeof(df[!, i]) == Vector{Union{Nothing,Bool}}
            if (df[!, i] .!= nothing) == 0
                println("Column $i is a vector of vector{Any}")
                remove_chk = true

            elseif sum(df[!, i] .== true) == 0
                println("Column $i is a vector of vector{Any}")
                remove_chk = true

            else
                println("***Column $i is not removed from Bool/Nothing check")
                remove_chk = true

            end

        elseif typeof(df[!, i]) == Vector{Union{Nothing,Int64}}
            if (df[!, i] .!= nothing) == 0
                println("Column $i is a vector of vector{Any}")
                remove_chk = true

            elseif sum(df[!, i] .== true) == 0
                println("Column $i is a vector of vector{Any}")
                remove_chk = true

            else
                println("***Column $i is not removed from Int64/Nothing check")
            end

        elseif typeof(df[!, i]) == Vector{Union{Nothing,Float64}}
            if (df[!, i] .!= nothing) == 0
                println("Column $i is a vector of vector{Any}")
                remove_chk = true

            elseif sum(df[!, i] .== true) == 0
                println("Column $i is a vector of vector{Any}")
                remove_chk = true
            else
                println("***Column $i is not removed from Float64/Nothing check")
            end
        elseif typeof(df[!, i]) == Vector{Union{Nothing,String}}
            if (df[!, i] .!= nothing) == 0
                println("Column $i is a vector of string but has all nothing values")
                remove_chk = true

            elseif sum(df[!, i] .!= "") == 0
                println("Column $i is a vector of string with all empty string values")
                remove_chk = true
            else
                println("***Column $i is not removed from String/Nothing check")
                #create new column with the name starts with i, and follow by _ and the sub column name
                # for j in names
            end
        elseif typeof(df[!, i]) == Vector{Union{Nothing,Vector{Any}}}
            if (df[!, i] .!= nothing) != 0
                println("Column $i is a vector of vector{Any} but has all nothing values")
                remove_chk = true
            end
        end
        #could add more remove criteria here. 
        if remove_chk
            select!(df, Not(i))
            println("Column $i is removed")
            push!(removed_columns, i)
        end
    end
    return removed_columns
end;

function replaceNothing(df::DataFrame)
    list_of_names = []
    for i in names(df)
        dummy = df[!, i]
        T = typeof(dummy)
        println(T)
        if T === Vector{Union{Nothing,String}}
            replace!(df[!, i], nothing => "")
            push!(list_of_names, i)
            println(i)
        elseif T == Vector{Union{Nothing,Bool}}
            replace!(df[!, i], nothing => false)
            push!(list_of_names, i)
            println(i)
        elseif T isa Vector{Union{Nothing,Int64}}
            replace!(df[!, i], nothing => 0)
            push!(list_of_names, i)
            println(i)
        elseif T isa Vector{Union{Nothing,Float64}}
            replace!(df[!, i], nothing => 0.0)
            push!(list_of_names, i)
            println(i)
        end
    end

    sort!(list_of_names)
    return list_of_names
end;


function splitNum_Unit(df::DataFrame, colname::String)
    # numerics = Vector{Float64}(0, length(df.gwp))
    numerics = zeros(Float64, length(df.gwp))
    units = repeat(["missing"], length(df.gwp))
    for i in 1:length(df[!, colname])
        if df[!, colname][i] === nothing
            numerics[i] = 0.0
            units[i] = "0 Recheck with original data"
            println("value changed at $colname, $i ")
        else
            s = split(df[!, colname][i], " ")

            if length(s) == 1
                if s[1][end-2:end] == "MPa"
                    numerics[i] = parse(Float64, s[1][1:end-3])
                    units[i] = "MPa"
                elseif s[1][end-2:end] == "psi"
                    numerics[i] = parse(Float64, s[1][1:end-3])
                    units[i] = "psi"
                else
                    println("Unit not found",s[1][end-2:end])
                end
            elseif length(s) == 2
                numerics[i] = parse(Float64, s[1])
                units[i] = s[2]
            elseif length(s) == 3
                numerics[i] = parse(Float64, s[1])
                units[i] = s[2]*" "*s[3]
            elseif length(s) == 4
                numerics[i] = parse(Float64, s[1])
                units[i] = s[2]*" "*s[3]*" "*s[4]
            else
                println("Error at $colname, $i")
                @error "Error at $colname, $i"
                break
            end
        end
    end
    return numerics, units
end;

function splitNum_Unit(df::DataFrame, colname::String, key::String)
    numerics = zeros(Float64, length(df.gwp))
    units = repeat(["missing"], length(df.gwp))
    for i in 1:length(df[!, colname])
        # println(i)
        if key ∉ keys(df[!, colname][i])
        elseif df[!, colname][i][key] === nothing 
            numerics[i] = 0.0
            units[i] = "0 Recheck with original data"
            println("value changed at $colname, $i ")
        else
            s = split(df[!, colname][i][key], " ")

            if length(s) == 1
                if s[1][end-2:end] == "MPa"
                    numerics[i] = parse(Float64, s[1][1:end-3])
                    units[i] = "MPa"
                elseif s[1][end-2:end] == "psi"
                    numerics[i] = parse(Float64, s[1][1:end-3])
                    units[i] = "psi"
                else
                    println("Unit not found",s[1][end-2:end])
                end
            elseif length(s) == 2
                numerics[i] = parse(Float64, s[1])
                units[i] = s[2]
            else
                println("length of s is not 1 or 2")
                print("HELP!!!")
                @error "HELP!!!"
            end
        end
    end
    return numerics, units
end;



function plot_country(df::DataFrame, country::String; savefig::Bool = true)
	f = Figure(resolution=(1200, 800))
	ax = Axis(f[1, 1], xlabel="Strength [MPa]", ylabel="GWP [kgCO2e/kg]")
	ax.title = "Strength vs GWP ($country)"
	ax.titlesize = 40
	xmax = maximum(df[!, "strength [MPa]"])
	ymax = maximum(df[!, "gwp_per_kg [kgCO2e/kg]"])
	if size(df)[1] < 10
		ax.xticks = 0:1:xmax
		ax.yticks = 0:0.01:ymax
	else
		ax.xticks = 0:10:xmax
		ax.yticks = 0:0.05:ymax
	end
	ax.xticks = 0:10:xmax
	ax.xlabelsize = 30
	ax.ylabelsize = 30
	scatter!(ax, df[!, "strength [MPa]"], df[!, "gwp_per_kg [kgCO2e/kg]"], color=:blue, markersize=20)
	f
	if savefig
		save("Plot_by_countries/$country.png", f)
		println("File save to Plot_by_countries/$country.png")
	end
	println("Done!")
	return f
end;

function plot_country(df::DataFrame, country::String, model::Chain; savefig::Bool = true)
	f = Figure(resolution=(1200, 800))
	ax = Axis(f[1, 1], xlabel="Strength [MPa]", ylabel="GWP [kgCO2e/kg]")
	ax.title = "Strength vs GWP ($country)"
	ax.titlesize = 40
	xmax = maximum(df[!, "strength [MPa]"])
	ymax = maximum(df[!, "gwp_per_kg [kgCO2e/kg]"])
	if size(df)[1] < 10
		ax.xticks = 0:1:xmax
		ax.yticks = 0:0.01:ymax
	else
		ax.xticks = 0:10:xmax
		ax.yticks = 0:0.05:ymax
	end

	ax.xlabelsize = 30
	ax.ylabelsize = 30
	scatter!(ax, df[!, "strength [MPa]"], df[!, "gwp_per_kg [kgCO2e/kg]"], color=:blue, markersize=20)
    xval = collect(10:0.1:xmax)
    xval_ = [ [x] for x in xval]
	lines!(ax, xval, [x[1] for x in model.(xval_)], color=:red, linewidth=3)
    f
	if savefig
        name = "$country"*"withSur.png"
		save("Plot_by_countries/"*name, f)
		println("File save to Plot_by_countries/$country.png")
	end
	println("Done!")
	return f
end;

function constructModels()
    # N1 = Chain(Dense(1, 1)) #need 10000 epoch

    N2_1 = Chain(Dense(1, 10, sigmoid), Dense(10, 1)) #need less than 5000 epoch
    # N2_2 = Chain(Dense(1, 10, relu), Dense(10, 1))
    N2_3 = Chain(Dense(1, 10, tanh), Dense(10, 1))

    N3_1 = Chain(Dense(1, 10), Dense(10, 10, sigmoid), Dense(10, 1))
    # N3_2 = Chain(Dense(1, 10), Dense(10, 10, relu), Dense(10, 1))
    N3_3 = Chain(Dense(1, 10), Dense(10, 10, tanh), Dense(10, 1))

    models = [N2_1, N2_3, N3_1, N3_3]
    model_names = [ "2_sigmoid" , "2_tanh", "3_sigmoid", "3_tanh"]
    return models, model_names
end

#normalize dataset
function normalize_data(data, x_max, x_min, y_max, y_min)
    data[:, 1] = (data[:, 1] .- x_min) ./ (x_max - x_min)
    data[:, 2] = (data[:, 2] .- y_min) ./ (y_max - y_min)
    return data
end

function un_normalize_data(data, x_max, x_min, y_max, y_min)
    data[:, 1] = data[:, 1] .* (x_max - x_min) .+ x_min
    data[:, 2] = data[:, 2] .* (y_max - y_min) .+ y_min
    return data
end


function plot_loss(save_model, save_loss, m_names ; ftitle = "Model loss vs Epoch")

    f_loss = Figure(resolution=(1200, 800))
    ax_loss = Axis(f_loss[1, 1], xlabel="Epoch", ylabel="Loss", 
                    yscale=log10, xscale = log10, 
                    title = ftitle,
                    xlabelsize = 30,
                    ylabelsize = 30,
                    titlesize  = 40)
#loop and plot all the models
    for i in eachindex(save_model)
        model = save_model[i]
        name = m_names[i]

        # design variables are fc′
        # assign model into function
        f2e = x -> sqrt.(x) #normalized modulus
        f2g = x -> model([x])[1] #will have to broadcast later.
        save_func_e[i] = deepcopy(f2e)
        save_func_g[i] = deepcopy(f2g)
    

        #get line type
        # line_type = :solid
        # println(string(name[1]))
        w = 3
        if string(name[1]) == "1"
        col = :black
        line_type = :solid
    elseif string(name[1]) == "2" 
        col = :red
        if string(name[end]) == "d"
            line_type = :solid
        elseif string(name[end]) == "u"
            line_type = :dot
        elseif string(name[end]) == "h"
            line_type = :dash
        end

    elseif string(name[1]) == "3"
        col = :blue
        if string(name[end]) == "d"
            line_type = :solid
        elseif string(name[end]) == "u"
            line_type = :dot
        elseif string(name[end]) == "h"
            line_type = :dash
        end
    end

	lines!(ax_func, xval, [x[1] for x in model.(xval_)], color=col, linestyle= line_type, linewidth= w, label = name)
    # lines!(ax_func, range_fc′, f2g(range_fc'), markersize=7.5, color=col, linestyle = line_type, label = name)
    lines!(ax_loss, save_loss[i], markersize=7.5, color=col, linestyle = line_type, label = name, linewidth = 5)
    # lines!(ax_loss, save_test_loss[i], markersize=7.5, color=col, linestyle = line_type, label = "test_"*name, linewidth = 2)

end
f_loss[1, 2] = Legend(f_loss, ax_loss, "Model", framevisible = false)
return f_loss
end



function get_nearest(train_data, f, dis)
    #get nearest neighours

    # f is a function 
    #this is for opt, so pick the uppep
    points = train_data
    x_data = Vector{Float32}(undef, 0)
    y_data = Vector{Float32}(undef, 0)
    for i =1:size(points)[1]
        x = points[i,1]
        y = points[i,2]
        if abs(y- f(x)) < dis
            push!(x_data, x)
            push!(y_data, y)
        end
    end

    #plot the original data
    f1 = Figure(resolution = (600, 600))
    ax1 = Axis(f1[1,1])
    scatter!(ax1, points[:,1], points[:,2], label = "original data" , color = :blue)

    #plot the data that is close to the function
    scatter!(ax1, x_data, y_data, label = "data close to function", color = :red)
    lines!(ax1 , 12:0.2:80, f.(12:0.2:80), label = "function", color = :green)
    # lines!(ax1, 12:0.2:80, f.(12:0.2:80) .+ dis, label = "function + dis", color = :black)
    #add legend
    f1[1,2] = Legend(f1, ax1, framevisible = false)

    data = hcat(x_data, y_data); # data is a 2 x n matrix
    return f1 , data
end

function plot_loss_func(save_model,save_loss, data, m_names = m_names; ftitle = "" , x_total = x_total, y_total = y_total)

    # save_func_e = Vector{Function}(undef, length(save_model))
    save_func_g = Vector{Function}(undef, length(save_model))

    f_func = Figure(resolution=(1200, 800))
    ax_func = Axis(f_func[1, 1], xlabel="Concrete strength [MPa]", ylabel="GWP [kgCO2e/kg]", title="Prediction"*" "*ftitle,
    xlabelsize = 30,
    ylabelsize = 30,
    titlesize  = 40,
    xticks = 0:10:100,
    yticks = 0:0.05:0.5)


    f_loss = Figure(resolution=(1200, 800))
    ax_loss = Axis(f_loss[1, 1], xlabel="Epoch", ylabel="Loss", title="Loss"*" "*ftitle,
    yscale = log10,
    xscale = log10,
    xlabelsize = 30,
    ylabelsize = 30,
    titlesize  = 40)

    ##===============================

    xmax = maximum(data[:,1])
    xmin = minimum(data[:,1])
# ymax = maximum(data[:,2])
# ymin = minimum(data[:,2])

    xval = collect(xmin:0.1:xmax)
    xval_ = [ [x] for x in xval] # for putting in model evaluation

    for i in eachindex(save_model)
        model = save_model[i]
        name = m_names[i]

        # design variables are fc′
        # assign model into function
        f2g = x -> model([x])[1] #will have to broadcast later.

        save_func_g[i] = deepcopy(f2g)
    
        #get line type
        # line_type = :solid
        # println(string(name[1]))
        w = 3
        if string(name[1]) == "1"
            col = :black
            line_type = :solid
        elseif string(name[1]) == "2" 
            col = :red
            if string(name[end]) == "d"
                line_type = :solid
            elseif string(name[end]) == "u"
                line_type = :dot
            elseif string(name[end]) == "h"
                line_type = :dash
            else
                @error "line type not defined"
            end
        elseif string(name[1]) == "3"
            col = :blue
            if string(name[end]) == "d"
                line_type = :solid
            elseif string(name[end]) == "u"
                line_type = :dot
            elseif string(name[end]) == "h"
                line_type = :dash
            else
                @error "line type not defined"
            end
        else
            @error "line type not defined"
        end

	lines!(ax_func, xval, [x[1] for x in model.(xval_)], color=col, linestyle= line_type, linewidth= w, label = name)
    lines!(ax_loss, save_loss[i], markersize=7.5, color=col, linestyle = line_type, label = name, linewidth = 5)
    # lines!(ax_loss, save_test_loss[i], markersize=7.5, color=col, linestyle = line_type, label = "test_"*name, linewidth = 2)
    
    end
    f_loss[1, 2] = Legend(f_loss, ax_loss, "Model", framevisible = false)
    f_func[1 ,2] = Legend(f_func, ax_func, "Model", framevisible = false)
    scatter!(ax_func, x_total, y_total, color=:gray) #plot all data

    return f_loss, f_func, save_func_g
end
