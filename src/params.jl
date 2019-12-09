module params

using Pkg.TOML
using DSP

export MicOptics, SimulationParams, Constants, ParameterSet, Space, MicroLensArray, LightFieldSimulation

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

    function MicOptics(optics::Dict{String,Any}, fml::Float64)

        ftl     = pop!(optics, "ftl")
        M       = pop!(optics, "M")
        NA      = pop!(optics, "NA")
        a0      = pop!(optics, "a0")
        b0      = pop!(optics, "b0")
        lambda  = pop!(optics, "lambda")
        n       = pop!(optics, "n")
        fobj    = ftl / M

        if b0 > fml
            d = b0
        else
            d = fml
        end

        new(ftl, fobj, M, NA, a0, b0, lambda, n, d)
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

    function SimulationParams(simparams::Dict{String,Any}, pitch::Float64)

        vpix    = pop!(simparams, "vpix")
        osr     = pop!(simparams, "osr")
        zmax    = pop!(simparams, "zmax")
        zmin    = pop!(simparams, "zmin")
        zstep   = pop!(simparams, "zstep")

        if mod(vpix, 2) == 0
            error("Invalid number of virtual pixels--must be odd number.")
        end

        subvpix = vpix * osr
        pixelpitch = pitch / vpix
        subpixelpitch = pixelpitch / osr
        new(vpix, osr, zmax, zmin, zstep, subvpix, pixelpitch, subpixelpitch)
        
    end
end

struct Constants

    k::Float64
    k0::Float64
    alpha::Float64

    function Constants(opt::MicOptics)

        k = 2 * pi * opt.n / opt.lambda
        k0 = 2 * pi * 1 / opt.lambda
        alpha = asin(opt.NA / opt.n)

        new(k, k0, alpha)
    end
end

struct Space
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    xlen::Int64
    ylen::Int64
    zlen::Int64
    center::Int64
    function Space(x = [0.0], y = [0.0], z = [0.0])
        xlen = length(x)
        ylen = length(y)
        zlen = length(z)
        center = cld(xlen, 2)
        new(x, y, z, xlen, ylen, zlen, center)
    end
end
struct ParameterSet
    opt::MicOptics
    sim::SimulationParams
    con::Constants
end

struct MicroLensArray

    pitch::Float64
    fml::Float64
    array::Array{ComplexF64,2}
    space::Space

    function MicroLensArray(pitch::Float64, fml::Float64, mlaspace::Space, img::Space, par::ParameterSet)
        #center = findfirst(img.x .== 0)
        mllen = mlaspace.xlen
        
        allcenters = vcat(img.center:-mllen:1, (img.center + mllen):mllen:img.xlen)
        sort!(allcenters) 
    
        a = zeros(ComplexF64, mllen, mllen)
    
        l2norm = mlaspace.x.^2 .+ mlaspace.y'.^2
        a .= exp.(-im * par.con.k / (2 * fml) .* l2norm);
    
        b = zeros(ComplexF64, img.xlen, img.ylen)
        b[allcenters, allcenters] .= 1
    
        #TODO: This is messy. Find a cleaner way to do it? works for now...
        c = conv(b, a)
        border = fld((size(c, 1) - img.xlen), 2) + 1
        crop = Int.(border:(img.xlen + (border-1)))
    
        new(pitch, fml, c[crop, crop], mlaspace)
    end
end


struct LightFieldSimulation
    par::ParameterSet
    mla::MicroLensArray
    img::Space
    obj::Space
end

end # module
