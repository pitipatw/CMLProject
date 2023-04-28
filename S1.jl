using Surrogates
num_samples = 10
lb = 0.0
ub = 10.0

#Sampling
x = sample(num_samples,lb,ub,SobolSample())
f = x-> log(x)*x^2+x^3
y = f.(x)

#Creating surrogate
alpha = 2.0
n = 6
my_lobachevsky = LobachevskySurrogate(x,y,lb,ub,alpha=alpha,n=n)

#Approximating value at 5.0
value = my_lobachevsky(5.0)

#Adding more data points
surrogate_optimize(f,SRBF(),lb,ub,my_lobachevsky,UniformSample())

#New approximation
value = my_lobachevsky(5.0)


