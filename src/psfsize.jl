module psfsize

using LightField.params
using LightField.psf
export calcsize

function calcsize(p::ParameterSet, objspace::Space)

    testline = collect((0:1:(p.sim.subvpix*20)) .* p.sim.subpixelpitch)
    x3max = maximum(abs.(objspace.z))

    lineprojection = psfline(testline, x3max, p.opt.M, p.con.k, p.con.alpha,
                                            p.opt.fobj, p.opt.lambda, p.opt.a0)

    imgspace = makeimgspace(lineprojection, p)

    return imgspace
end

function psfline(x₁testline::Vector{Float64}, x₃max::Float64, M::Float64,
                    k::Float64, α::Float64, fₒ::Float64, λ::Float64,a₀::Float64)

    l²norm² = abs.(x₁testline) ./ M
    v = l²norm² .* (k*sin(α))
    u = 4k * x₃max * (sin(α/2)^2)
    Kₒ = M/(fₒ*λ)^2 * exp(-im*u/(4(sin(α/2)^2)))

    linearpsf = complex(zeros(length(v)))

    Threads.@threads for i in 1:length(v)
        @inbounds linearpsf[i] = integratePSF(v[i], u, a₀, α) * Kₒ
    end

    linearpsfmag = Float64.( abs2.(linearpsf) ./ maximum(abs2.(linearpsf)) )

    return linearpsfmag
end

function makeimgspace(psfline::Vector{Float64}, p::ParameterSet)

    # Set a threshold where the value of the psfline reaches ~0
    outArea = psfline .< 0.01

    # This will likely never happen but good flag to check if we do huge volumes
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
