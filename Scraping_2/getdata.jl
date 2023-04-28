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
function get_data(cat::String, page::Int64; token::String = token, mode::Int64 = 0)
    if mode == 0
        res = HTTP.request( "GET",
        "https://buildingtransparency.org/api/$cat/?page_number=$page", #materials?page_number=2",
        ["Authorization" => "Bearer "*"$token"]
        )
    else
        println("Mode not found")
        res = 0.0
            
    end
    return res
end

#test run
res = get_data("", page)

res = 0
cat = "materials"
page = 1 #start from page 1
total_pages = 0
filepath = joinpath(@__DIR__,"all_files/")
msg = 0 

# # at test run 
# res = get_data(cat, page)
# response_text = String(res.body)
# filename = "ECS_page_" * string(0) * ".json"
# msg = JSON.parse(response_text)

@time while page != total_pages
    if page == 1 
        res = get_data(cat, page)
        response_text = String(res.body)
        filename = "ECS_page_" * string(page) * ".json"
        total_pages = parse(Int64,(res["X-Total-Pages"]))
        println(total_pages)

        msg = JSON.parse(response_text)
        #write a to a json file
        println(filepath*filename)
        open(filepath*filename, "w") do f
            JSON.print(f, msg)
        end
        # open(filepath*filename, "w") do f
        #     write(f, msg)
        # end
        page += 1
    else
        res = get_data(cat, page)
        response_text = String(res.body)
        filename = "ECS_page_" * string(page) * ".json"

        msg = JSON.parse(response_text)
        #write a to a json file
        println(filepath*filename)
        open(filepath*filename, "w") do f
            JSON.print(f, msg)
        end
        # open(filepath*filename, "w") do f
        #     write(f, msg)
        # end
        page += 1
        res = nothing #clear the memory
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