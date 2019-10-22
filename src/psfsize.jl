module psfsize

using ..params
using ..psf
using FFTW
export calcsize

function calcsize(p::ParameterSet, obj::Space)
    # we set an arbitrarily high line length to test across
    testline = collect((0:1:(p.sim.subvpix*20)) .* p.sim.subpixelpitch)
    x3max = maximum(abs.(obj.z))

    if p.opt.a0 > zero(Float64)
        lineprojection = Array{Float64,2}(undef, length(testline), obj.zlen)
        for i in 1:obj.zlen
            lineprojection[:,i] .= psfline(testline, obj.z[i], p)
        end
    
    
    else 
        lineprojection = psfline(testline, x3max, p)
    end
    psflinemag::Vector{Float64} =  abs2.(lineprojection) ./ maximum(abs2.(lineprojection))    
    imgspace = makeimgspace(lineprojection, p)

    return imgspace
end

function psfline(x₁line::Vector{Float64}, x₃::Float64, p::ParameterSet)

    l²norm² = abs.(x₁line) ./ p.opt.M
    v = l²norm² .* (p.con.k*sin(p.con.alpha))
    u = 4*p.con.k * x₃ * (sin(p.con.alpha/2)^2)
    Kₒ = p.opt.M/(p.opt.fobj*p.opt.lambda)^2 * exp(-im*u/(4(sin(p.con.alpha/2)^2)))

    psfline = Array{ComplexF64,1}(undef,length(v))

    Threads.@threads for i in 1:length(v)
        @inbounds psfline[i] = integratePSF(v[i], u, p.opt.a0, p.con.alpha) * Kₒ
    end

    return psflinemag
end

function makeimgspace(psfline::Vector{Float64}, p::ParameterSet)

    outArea = psfline .< 0.01
    if sum(outArea) == 0
        error("Estimated PSF size exceeds the limit")
    end

    sizeref = cld(findfirst(outArea)[1],p.sim.subvpix)

    if p.opt.a0 > 0
        """ Gives the number of supersampled pixels across the image. Note padding
        of 1 extra microlens after ceiling value above."""
        halfwidth = max( p.sim.subvpix*(sizeref + 1), 12*p.sim.subvpix)
        # 12 is arbitrary; needs to be high bc img width can be larger near origin
        # than it is at x3max if using a0 displacement
    else
        halfwidth = max( p.sim.subvpix*(sizeref + 1), 2*p.sim.subvpix)
        # simple linear scaling works fine with classic LFM
    end

    "Create X-Y image space based on halfwidth."
    x =(-halfwidth:1:halfwidth) * p.sim.subpixelpitch
    y =(-halfwidth:1:halfwidth) * p.sim.subpixelpitch
    imgspace = Space(x,y)

    return imgspace
end

end # module
