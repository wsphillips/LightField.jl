module params

using TOML

export setup

struct MicOptics
    ftl::Float64
    M::Float64
    NA::Float64
    a0::Float64
    b0::Float64
    lambda::Float64
    n::Float64
    function MicOptics(optics::Dict{AbstractString,Any})
        ftl     = pop!(optics, "ftl")
        M       = pop!(optics, "M")
        NA      = pop!(optics,"NA")
        a0      = pop!(optics,"a0")
        b0      = pop!(optics,"b0")
        lambda  = pop!(optics,"lambda")
        n       = pop!(optics,"n")
        new(ftl,M,NA,a0,b0,lambda,n)
    end
end

struct MicroLensArray
    pitch::Float64
    fml::Float64
    function MicroLensArray(mla::Dict{AbstractString,Any})
        pitch   = pop!(mla, "pitch")
        fml     = pop!(mla, "fml")
        new(pitch,fml)
    end
end

struct SimulationParams
    vpix::Int64
    osr::Int64
    zmax::Float64
    zmin::Float64
    zstep::Float64
    function SimulationParams(simparams::Dict{AbstractString,Any})
        vpix    = pop!(simparams,"vpix")
        osr     = pop!(simparams,"osr")
        zmax    = pop!(simparams,"zmax")
        zmin    = pop!(simparams, "zmin")
        zstep   = pop!(simparams, "zstep")
        new(vpix,osr,zmax,zmin,zstep)
    end
end

function setup(filename::AbstractString)
    config = TOML.parsefile(filename)
    opt = MicOptics(pop!(config,"optics"))
    mla = MicroLensArray(pop!(config,"mla"))
    sim = SimulationParams(pop!(config,"simparams"))
    return (opt, mla, sim)
end

end
