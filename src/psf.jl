module psf

using FFTW
using ..params
import SpecialFunctions.besselj
import FastGaussQuadrature.gausslegendre
import LinearAlgebra.Symmetric


export integratePSF, fresnelH, originPSFproj

# NOTE: Number of quadrature nodes can be tuned for speed or accuracy
const (nodes, weights) = gausslegendre(100)

function intervalchange(x::Float64,a::Float64,b::Float64)
    newx = x*(b-a)/2 + (a+b)/2
    return newx
end

function lfmPSF(θ::Float64, α::Float64, v::Float64, u::Float64, a₀::Float64)
    if a₀ > zero(Float64)
        # PSF as given by Li et al. 2018. Used when MLA is offset from NIP by distance a₀
        I = sqrt(cos(θ)) * besselj(0,v*sin(θ)/sin(α)) * exp((-im*u*sin(θ/2)^2)/(2*sin(α/2)^2))*sin(θ)
    else
        # PSF for classic LFM
        I = sqrt(cos(θ)) * besselj(0, v*sin(θ)/sin(α)) * exp((im*u*sin(θ/2)^2)/(2*sin(α/2)^2))*sin(θ)
    end
    return I
end

function integratePSF(v::Float64, u::Float64, a₀::Float64, α::Float64)
    θ = intervalchange.(nodes, 0.0, α)
    integral = (α/2) * sum(lfmPSF.(θ, α, v, u, a₀) .* weights)
    return integral
end

function fresnelH(f0::Array{Complex{Float64},N}, par::ParameterSet, z::Float64) where N

    Nx = size(f0,1)
    f0length = par.sim.subpixelpitch*Nx

    # Generates spatial frequency range at 256-bit precision to avoid inexact error
    truestep = BigFloat(one(BigFloat)/f0length)
    endpt = BigFloat(one(BigFloat)/(2*par.sim.subpixelpitch))
    spfreq = range(-endpt, stop=endpt-truestep, length=Nx)

    # Setup frequency axes
    fx = spfreq'
    fy = reverse(spfreq, dims=1)
    if N == 2
        FXFY = fx.^2 .+ fy.^2
    elseif N == 1
        FXFY = fy.^2
    end # for correctness we could include error throw if wrong array dims

    # Fresnel propagation func. (see: Computational Fourier Optics p.54-55,63)
    H = exp(im*par.con.k0*z).*exp.((-im*pi*par.opt.lambda*z) .* FXFY)
    # Shift H for later fourier space calc + convert back to 64-bit precision
    return ComplexF64.(fftshift(H))
end

struct OriginPSF
    patternstack::Array{Array{Complex{Float64},2},1}
    originimgs::Array{Float64,3}    

    function OriginPSF(lf::LightFieldSimulation)
        zmax = maximum(lf.obj.z)
        patternstack = Array{Array{Complex{Float64},2},1}(undef, lf.obj.zlen)
        originimgs = Threads.@spawn zeros(ComplexF64, lf.img.xlen, lf.img.ylen, lf.obj.zlen)

        Threads.@threads for p in 1:lf.obj.zlen
            imgrefsize = cld((lf.img.xlen * abs(lf.obj.z[p])), zmax)
            halfwidth = max(imgrefsize * lf.par.sim.subvpix, 2 * lf.par.sim.subvpix)
            centerarea = max((lf.img.center - halfwidth + 1), 1):1:min((lf.img.center + halfwidth - 1), lf.img.xlen)
            patternstack[p] = Array{Complex{Float64},2}(undef, length(centerarea), length(centerarea))
        end

        new(patternstack, fetch(originimgs))
    end
end

function originPSFproj(lf::LightFieldSimulation)

    #(pattern_stack, originimgs) = originPSFalloc(obj, img, par)
    psf = OriginPSF(lf)
    originPSFproj(psf, lf)

    #This is what happens when Shu Jia lab super resolution is applied
    if lf.par.opt.a0 > 0.0
        steps::Int64 = 10
        stepz = lf.par.opt.a0/steps
        Ha0 = fresnelH(psf.originimgs[:,:,1], lf.par, stepz)
        itrfresnelconv!(psf.originimgs, Ha0, steps, lf.obj)
    end

    return psf.originimgs
end

#= TODO: Clean this up
function originPSFalloc(obj::Space, img::Space, par::ParameterSet)
    zmax = maximum(obj.z)
    pattern_stack = Array{Array{Complex{Float64},2},1}(undef, obj.zlen)
    originimgs = complex(zeros(img.xlen,img.ylen,obj.zlen))
    Threads.@threads for p in 1:obj.zlen
        sizeref = cld((img.xlen*abs(obj.z[p])),zmax)
        halfwidth =  max(sizeref*par.sim.subvpix, 2*par.sim.subvpix)
        centerarea = max((img.center - halfwidth + 1) , 1):1:min((img.center + halfwidth - 1), img.xlen)
        pattern_stack[p] = complex(zeros(Float64, length(centerarea), length(centerarea)))
    end

    return (pattern_stack, originimgs)
end
=#


function unfold(I1::Array{Complex{Float64},2}, pattern::Array{Complex{Float64},2})
    middle::Int64 = cld(size(pattern,1),2)
    pattern[1:middle, 1:middle] .= I1[:,:]
    pattern[1:middle, middle:end] .= reverse(I1, dims=2)
    pattern[middle:end, 1:end] .= reverse(pattern[1:middle, 1:end], dims=1)
    return pattern
end

#TODO: cleanup
function originPSFproj(psf::OriginPSF, lf::LightFieldSimulation)
    zmax = maximum(obj.z)
    vscalar = lf.par.con.k * sin(lf.par.con.alpha)
    uscalar = 4 * lf.par.con.k * (sin(lf.par.con.alpha / 2)^2)
    Threads.@threads for j in 1:obj.zlen

        sizeref = cld((img.xlen * abs(obj.z[j])),zmax)
        halfwidth::Int64 =  max(sizeref * par.sim.subvpix, 2 * par.sim.subvpix)
        centerarea::Array{Int64,1} = max((img.center - halfwidth + 1), 1):1:min((img.center + halfwidth - 1), img.xlen)        
        
        triangle = falses(length(centerarea[1]:img.center), length(centerarea[1]:img.center))

        for x in 1:size(triangle,1)
            triangle[1:x, x] = true
        end

        xl2 = ((img.x[centerarea[1]:img.center]'.^2  .+ img.y[centerarea[1]:img.center].^2) .^0.5) ./ par.opt.M

        v = xl2 .* vscalar
        u = obj.z[j] * uscalar
        Koi = par.opt.M / ((par.opt.fobj* par.opt.lambda)^2) * exp(-im * u / (4 * (sin(par.con.alpha / 2)^2)))

        integral = zeros(ComplexF64,size(v))
        integral[triangle] .= integratePSF.(v[triangle], u, par.opt.a0, par.con.alpha)
        integral[triangle] .= integral[triangle] .* Koi
        copyto!(integral, Symmetric(integral))

        psf.patternstack[j] .= unfold(integral, psf.patternstack[j])
        originimgs[centerarea, centerarea, j] .= psf.patternstack[j][:,:]
    end
    return
end

function itrfresnelconv!(originimgs::Array{Complex{Float64},3}, Ha0::Array{Complex{Float64},2}, steps::Int64, obj::Space)
    f0 = originimgs[:,:,1]
    p = plan_fft!(f0, [1,2])

    Threads.@threads for h in 1:length(obj.z)
        f0 .= originimgs[:,:,h]
        # Fourier space computation
        p*f0                   # Applies fft in place
        f0 .= f0.*(Ha0.^steps) # Multiply by transfer function each multiplication is an incremental proj
        p\f0                   # Applies ifft in place
        originimgs[:,:,h] .= f0
    end
    return
end

end # psf module end
