using CSV , DataFrames
using Surrogates
using Makie, GLMakie

#load csv file into a dataframe called df
df = CSV.read("readytorun2.csv", DataFrame)

df[!,"country"]
df_IN = df[df[!,"country"] .== "IN",:]
df_US = df[df[!,"country"] .== "US",:]
df_CA = df[df[!,"country"] .== "CA",:]
df_AU = df[df[!,"country"] .== "AU",:]
df_NZ = df[df[!,"country"] .== "NZ",:]
df_SG = df[df[!,"country"] .== "SG",:]


#take the strength [Mpa] and gwp_per_kg [kgCO2e/kg] for the train and test dataset. 



lb = 0.0
ub = 50.0

#Sampling
x = collect(df_IN[!,"strength [MPa]"])
y = collect(df_IN[!,"gwp_per_kg [kgCO2e/kg]"])
#get x train rancomly from X
x_train = x[rand(1:length(x), 100)]
#get y train rancomly from Y
y_train = y[rand(1:length(y), 100)]




#Creating surrogate
alpha = 2.0
n = 2
my_lobachevsky = LobachevskySurrogate(x,y,lb,ub,alpha=alpha,n=n)
#Approximating value at 5.0



value = my_lobachevsky(50.0)

#plot the original data and randomly sample data
x_rand = sample(10,lb,ub,SobolSample())
y_rand = my_lobachevsky.(x_rand)
fig = Figure(resolution = (1200,800))
ax1 = Axis(fig, xlabel = "strength [MPa]", ylabel = "gwp_per_kg [kgCO2e/kg]")
scatter(x,y,label="original data")
scatter!(x_rand,y_rand,label="randomly sampled data")


#Adding more data points
surrogate_optimize(f,SRBF(),lb,ub,my_lobachevsky,UniformSample())

#New approximation
value = my_lobachevsky(5.0)
