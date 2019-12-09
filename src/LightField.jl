module LightField

include("params.jl")
include("psf.jl")
include("psfsize.jl")

include("projection.jl")

using .psf, .psfsize, .params, .projection
using Pkg.TOML

function setup(filename::String)

    config = TOML.parsefile(filename)
    mladict = pop!(config, "mla")
    pitch = pop!(mladict, "pitch")
    fml = pop!(mladict, "fml")


    opt = MicOptics(pop!(config, "optics"), fml)
    sim = SimulationParams(pop!(config, "simparams"), pitch)
    con = Constants(opt)

    xobj::Vector{Float64} = (sim.pixelpitch / opt.M) .* collect(-fld(sim.vpix,2):1:fld(sim.vpix,2))
    yobj::Vector{Float64} = (sim.pixelpitch / opt.M) .* collect(-fld(sim.vpix,2):1:fld(sim.vpix,2))
    zobj::Vector{Float64} = sim.zmin:sim.zstep:sim.zmax

    xml::Vector{Float64} = collect(-fld((sim.subvpix - 1), 2):1:fld((sim.subvpix - 1), 2)) .* sim.subpixelpitch
    yml::Vector{Float64} = collect(-fld((sim.subvpix - 1), 2):1:fld((sim.subvpix - 1), 2)) .* sim.subpixelpitch
    obj = Space(xobj, yobj, zobj)
    mlaspace = Space(xml, yml)
    par = ParameterSet(opt, sim, con)
    img = calcsize(par,obj)
    mla = MicroLensArray(pitch, fml, mlaspace, img, par)
    lf = LightFieldSimulation(par, mla, img, obj)
    return lf
end



export setup, propagate


end # module
