module psfsize

using LightField.params
using LightField.psf
export calcsize

function calcsize(p::ParameterSet, obj::Space)

    testline = collect((0:1:(p.sim.subvpix*20)) .* p.sim.subpixelpitch)
    x3max = maximum(abs.(obj.z))

    lineprojection = psfline(testline, x3max, p)
    imgspace = makeimgspace(lineprojection, p)

    return imgspace
end

function psfline(x₁testline::Vector{Float64}, x₃max::Float64, p::ParamaterSet)

    l²norm² = abs.(x₁testline) ./ p.opt.M
    v = l²norm² .* (p.con.k*sin(p.con.alpha))
    u = 4*p.con.k * x₃max * (sin(p.con.alpha/2)^2)
    Kₒ = p.opt.M/(p.opt.fobj*p.opt.lambda)^2 * exp(-im*u/(4(sin(p.con.alpha/2)^2)))

    linearpsf = complex(zeros(length(v)))

    Threads.@threads for i in 1:length(v)
        @inbounds linearpsf[i] = integratePSF(v[i], u, p.opt.a0, p.con.alpha) * Kₒ
    end

    linearpsfmag = Float64.( abs2.(linearpsf) ./ maximum(abs2.(linearpsf)) )

    return linearpsfmag
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
