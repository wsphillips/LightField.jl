module LightField

include("params.jl")
include("psf.jl")
include("psfsize.jl")

using .psf, .psfsize, .params
export setup, calcsize


end # module
