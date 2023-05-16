using Pkg

# for Topology optimization
Pkg.add("TopOpt")
Pkg.add("LinearAlgebra")
Pkg.add("StatsFuns")
# for Data Visualization
Pkg.add("Makie")
Pkg.add("GLMakie")
# for Data Analysis
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Clustering")
#These are for the surrogate model
Pkg.add("Optimisers")
Pkg.add("Flux")
Pkg.add("Zygote")
Pkg.add("MLJ")
Pkg.add("SurrogatesFlux"), 
Pkg.add("Surrogates")
Pkg.add("Statistics")
Pkg.add("Random")
Pkg.add("Distributions")

Pkg.add("ProgressLogging")