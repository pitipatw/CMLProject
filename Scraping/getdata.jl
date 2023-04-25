using HTTP, JSON
#GOAL
#at the first loop, get the total number of pages
#get that by res["X-Total-Pages"]
#Each page will have 100 items, therefore loop from 1 to res["X-Total-Pages"]
#at each loop, get the data from the page
#save the data to a json file separate by the filename 
#filename = "ECS" * string(i) * ".json"

include("key.jl")
# Authorization: Bearer "$token"
# curl -H "Authorization: Bearer TX6wOlpw03TIaEI5TNyEW5tNddSzpN" https://buildingtransparency.org/api/epds
#get data from buildingtransparency.org/api/epds using the api api_key
function get_data(cat::String, pagenum::Int64)
    res = HTTP.request( "GET",
        "https://buildingtransparency.org/api/$cat/?page_number=$pagenum", #materials?page_number=2",
        ["Authorization" => "Bearer "*"$token"]
    )
    return res
end


cat = "materials"
page = 1 #start from page 1
total_pages = 0
filepath = joinpath(@__DIR__,"all_files/")
msg = 0 
while page != total_pages
    if page == 1 
        res = get_data(cat, page)
        response_text = String(res.body)
        filename = "ECS_page_" * string(page) * ".json"
        total_pages = parse(Int64,(res["X-Total-Pages"]))
        println(total_pages)

        msg = JSON.json(response_text)
        #write a to a json file
        println(filepath*filename)
        open(filepath*filename, "w") do f
            write(f, msg)
        end
    else
        println("break")
        break
        res = get_data()
        response_text = String(res.body)
        a = JSON.parse(response_text)
        a
        #write a to a json file
        open("data.json", "a") do io
            JSON.print(io, a)
        end
    end
end

res = get_data()
response_text = String(res.body)
a = JSON.parse(response_text)
a
#write a to a json file
open("data.json", "w") do io
    JSON.print(io, a)
end