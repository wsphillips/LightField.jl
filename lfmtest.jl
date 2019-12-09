
using Revise
using LightField
using Plots; gr()
cd("/home/wikphi/LightField.jl/")
lf = setup("configexample.toml")
@time result = propagate(lf)

normal = result[:,:,:,:,1] ./ maximum(result[:,:,:,:,1])
using FFTW
normal[normal .< (0.005*maximum(normal))] .= 0
heatmap(normal[:,:,13,27])
