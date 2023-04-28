using Surrogates
f = x -> exp(x)*x^2+x^3
lb = 0.0
ub = 10.0
x = sample(50,lb,ub,UniformSample())
y = f.(x)
p = 1.9
my_krig = Kriging(x,y,lb,ub,p=p)

#I want an approximation at 5.4
approx = my_krig(5.4)

#I want to find the standard error at 5.4
std_err = std_error_at_point(my_krig,5.4)