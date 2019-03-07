module LightField

include("params.jl")
include("psf.jl")
include("psfsize.jl")

using params, psf, psfsize

export setup, calcsize


end # module
