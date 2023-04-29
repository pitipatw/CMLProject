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
end

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
end


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
end

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
end