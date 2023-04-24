using HTTP, JSON
#GOAL
#at the first loop, get the total number of pages
#get that by res["X-Total-Pages"]
#Each page will have 100 items, therefore loop from 1 to res["X-Total-Pages"]
#at each loop, get the data from the page
#save the data to a json file separate by the filename 
#filename = "ECS" * string(i) * ".json"



include("key.jl")
Authorization: Bearer "$token"
# curl -H "Authorization: Bearer TX6wOlpw03TIaEI5TNyEW5tNddSzpN" https://buildingtransparency.org/api/epds
#get data from buildingtransparency.org/api/epds using the api api_key
function get_data()
    res = HTTP.request( "GET",
        "https://buildingtransparency.org/api/materials", #materials?page_number=2",
        ["Authorization" => "Bearer "*"$token"]
    )
    return res
end

res = get_data()
response_text = String(res.body)
a = JSON.parse(response_text)
a
#write a to a json file
open("data.json", "w") do io
    JSON.print(io, a)
end