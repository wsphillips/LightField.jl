module psfsize

using ..params
using ..psf
using FFTW
export calcsize

function calcsize(p::ParameterSet, obj::Space)
    # we set an arbitrarily high length to test across
    testrange = collect((0:1:(p.sim.subvpix*20)) .* p.sim.subpixelpitch)
    x3max = maximum(abs.(obj.z))

    if p.opt.a0 > zero(Float64)
        steps = 10
        stepz = p.opt.a0/steps
        lineproj = zeros(ComplexF64, length(testrange), obj.zlen)
        linemag = zeros(length(testrange), obj.zlen)
        Hline = fresnelH(lineproj[:,1], p, stepz)
        plan = plan_fft!(lineproj[:,1])
        for i in 1:obj.zlen
            lineproj[:,i] = psfline(testrange, obj.z[i], p)
            plan*lineproj[:,i]
            lineproj[:,i] = lineproj[:,i].*(Hline.^steps)
            plan\lineproj[:,i]
            linemag[:,i] = abs2.(lineproj[:,i]) ./ maximum(abs2.(lineproj[:,i]))
        end
        imgspace = makeimgspace(dropdims(reduce(max, linemag, dims=2); dims=2), p)
    else
        lineproj = psfline(testrange, x3max, p)
        linemag =  abs2.(lineproj) ./ maximum(abs2.(lineproj))
        imgspace = makeimgspace(linemag, p)
    end
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

    return psfline
end

function makeimgspace(psfline::Vector{Float64}, p::ParameterSet)
    outArea = psfline .< 0.01
    if sum(outArea) == 0
        error("Estimated PSF size exceeds the limit")
    end
    sizeref = cld(findfirst(outArea)[1],p.sim.subvpix)
    halfwidth = max( p.sim.subvpix*(sizeref + 1), 2*p.sim.subvpix)

    "Create X-Y image space based on halfwidth."
    x =(-halfwidth:1:halfwidth) * p.sim.subpixelpitch
    y =(-halfwidth:1:halfwidth) * p.sim.subpixelpitch
    imgspace = Space(x,y)

    return imgspace
end

end # module
