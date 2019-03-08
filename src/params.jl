module params

using TOML

export setup
export ParameterSet, Space

struct MicroLensArray

    pitch::Float64
    fml::Float64

    function MicroLensArray(mla::Dict{AbstractString,Any})

        pitch   = pop!(mla, "pitch")
        fml     = pop!(mla, "fml")

        new(pitch,fml)
    end
end

struct MicOptics

    ftl::Float64
    fobj::Float64
    M::Float64
    NA::Float64
    a0::Float64
    b0::Float64
    lambda::Float64
    n::Float64
    d::Float64

    function MicOptics(optics::Dict{AbstractString,Any}, mla::MicroLensArray)

        ftl     = pop!(optics, "ftl")
        M       = pop!(optics, "M")
        NA      = pop!(optics,"NA")
        a0      = pop!(optics,"a0")
        b0      = pop!(optics,"b0")
        lambda  = pop!(optics,"lambda")
        n       = pop!(optics,"n")
        fobj    = ftl / M

        if b0 > mla.fml
            d = b0
        else
            d = fml
        end

        new(ftl,fobj,M,NA,a0,b0,lambda,n,d)
    end
end

struct SimulationParams

    vpix::Int64
    osr::Int64
    zmax::Float64
    zmin::Float64
    zstep::Float64
    subvpix::Int64
    pixelpitch::Float64
    subpixelpitch::Float64

    function SimulationParams(simparams::Dict{AbstractString,Any}, mla::MicroLensArray)

        vpix    = pop!(simparams,"vpix")
        osr     = pop!(simparams,"osr")
        zmax    = pop!(simparams,"zmax")
        zmin    = pop!(simparams, "zmin")
        zstep   = pop!(simparams, "zstep")

        if mod(vpix,2)==0
            error("Invalid number of virtual pixels--must be odd number.")
        end

        subvpix = vpix*osr
        pixelpitch = mla.pitch / vpix
        subpixelpitch = pixelpitch / osr

        new(vpix,osr,zmax,zmin,zstep, subvpix, pixelpitch, subpixelpitch)
    end
end

struct Constants

    k::Float64
    k0::Float64
    alpha::Float64

    function Constants(opt::MicOptics)

        k = 2*pi*opt.n/opt.lambda
        k0 = 2*pi*1/opt.lambda
        alpha = asin(opt.NA/opt.n)

        new(k, k0, alpha)
    end
end

struct ParameterSet
    mla::MicroLensArray
    opt::MicOptics
    sim::SimulationParams
    con::Constants
end

function setup(filename::String)

    config = TOML.parsefile(filename)

    mla = MicroLensArray(pop!(config,"mla"))
    opt = MicOptics(pop!(config,"optics"), mla)
    sim = SimulationParams(pop!(config,"simparams"), mla)
    con = Constants(opt)
    par = ParameterSet(mla, opt, sim, con)

    # TODO: flesh out space definitions here
    # add halfwidth, centerpt? max?

    xobj = collect((sim.pixelpitch/opt.M) .* (-sim.vpix:1:sim.vpix))
    yobj = collect((sim.pixelpitch/opt.M) .* (-sim.vpix:1:sim.vpix))
    zobj = collect(sim.zmin:sim.zstep:sim.zmax)

    xml = collect((-(sim.subvpix-1)/2 : 1 : (sim.subvpix-1)/2) .*
                                                            sim.subpixelpitch)
    yml = collect((-(sim.subvpix-1)/2 : 1 : (sim.subvpix-1)/2) .*
                                                            sim.subpixelpitch)

    # ...then image space is defined by psfsize functions

    objspace = Space(xobj, yobj, zobj)
    mlaspace = Space(xml, yml)

    return (par, objspace, mlaspace)
end

struct Space
    # TODO: Decide if there should be more embedded parameters
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    xlen::Int64
    ylen::Int64
    zlen::Int64
    function Space(x=[0.0],y=[0.0],z=[0.0])
        xlen = length(x)
        ylen = length(y)
        zlen = length(z)

        new(x,y,z,xlen,ylen,zlen)
    end
end


end # module
