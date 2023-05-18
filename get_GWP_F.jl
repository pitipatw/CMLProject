using HTTP.WebSockets
using JSON



server = WebSockets.listen!("127.0.0.1", 2000) do ws
    for msg in ws
        open(joinpath(@__DIR__, "fc_input.json"), "w") do f
            write(f, msg)
        end
        fc = JSON.parse(msg)


        f2e = save_func_e[2]
        f2g = save_func_g[2]
        E = f2e(fc)
        GWP = f2g(fc)

        results = Dict
        results["E"] = E
        results["GWP"] = GWP
        msg = JSON.json(results)
        send(ws, msg) 
        println()

        savepath = joinpath(@__DIR__, "E_output.json")
        open(joinpath(@__DIR__,savepath), "w") do f
            write(f, msg)
        end
        println("json output file written Successfully")
    end
end



close(server)