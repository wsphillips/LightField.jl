module LightField

include("params.jl")
include("psf.jl")
include("psfsize.jl")
include("mla.jl")
include("projection.jl")

using .psf, .psfsize, .params, .mla, .projection 
using TOML

function setup(filename::String)

    config = TOML.parsefile(filename)
    mladict = pop!(config, "mla")
    pitch = pop!(mladict, "pitch")
    fml = pop!(mladict, "fml")


    opt = MicOptics(pop!(config, "optics"), fml)
    sim = SimulationParams(pop!(config, "simparams"), pitch)
    con = Constants(opt)

    xobj = collect((sim.pixelpitch / opt.M) .* (-sim.vpix:1:sim.vpix))
    yobj = collect((sim.pixelpitch / opt.M) .* (-sim.vpix:1:sim.vpix))
    zobj = collect(sim.zmin:sim.zstep:sim.zmax)

    xml = collect((-(sim.subvpix - 1) / 2:1:(sim.subvpix - 1) / 2) .*
                                                            sim.subpixelpitch)
    yml = collect((-(sim.subvpix - 1) / 2:1:(sim.subvpix - 1) / 2) .*
                                                            sim.subpixelpitch)
    obj = Space(xobj, yobj, zobj)
    mlaspace = Space(xml, yml)
    mla = MicroLensArray(pop!(config, "mla"))
    par = ParameterSet(opt, sim, con)

    # TODO: fix up the argument passing and then run calcml()
    img = calcsize(par,obj)

    lf = LightField(par, mla, img, obj)
    return lf
end



export setup, propagate


end # module
