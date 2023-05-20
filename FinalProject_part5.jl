using HTTP.WebSockets
using JSON

#run this with GH
f2g = f2g_IN

server = WebSockets.listen!("127.0.0.1", 2000) do ws
    for msg in ws
        open(joinpath(@__DIR__, "fc_input.json"), "w") do f
            #json file of fc'
            write(f, msg)
        end
        df = JSON.parse(msg)
        fc = df["fc"]
        GWP = f2g(fc)


        results = Dict
        results["GWP"] = GWP
        msg = JSON.json(results)
        send(ws, msg) 
        println("JSON sent Successfully")

        savepath = joinpath(@__DIR__, "GWP_output.json")
        open(joinpath(@__DIR__,savepath), "w") do f
            write(f, msg)
        end
        println("json output file written Successfully")
    end
end



close(server)