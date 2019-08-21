module psf

using FFTW
import SpecialFunctions.besselj
import FastGaussQuadrature.gausslegendre
import LinearAlgebra.Symmetric
import LightField.params.ParameterSet, LightField.params.Space
export intpsf, makeHmatrix, calcPSF

function intervalchange(x::Float64,a::Float64,b::Float64)
    newx = x*(b-a)/2 + (a+b)/2
    return newx
end

function lfmpsf(θ::Float64, α::Float64, v::Float64, u::Float64, a₀::Float64)
    if a₀ > 0.0
        # PSF for classic LFM
        I = sqrt(cos(θ)) * besselj(0, v*sin(θ)/sin(α)) * exp((im*u*sin(θ/2)^2)/(2*sin(α/2)^2))*sin(θ)
    else
        # PSF as given by Li et al. 2018. Used when MLA is offset from NIP by
        # distance a₀
        I = sqrt(cos(θ)) * besselj(0,v*sin(θ)/sin(α)) * exp((-im*u*sin(θ/2)^2)/(2*sin(α/2)^2))*sin(θ)
    end
    return I
end

function intpsf(v::Float64, u::Float64, a₀::Float64, α::Float64)
    # NOTE: Number of quadrature nodes can be tuned for speed or accuracy
    (x, weights) = gausslegendre(100)
    θ = intervalchange.(x, 0.0, α)
    integral = (α/2) * sum(lfmpsf.(θ, α, v, u, a₀) .* weights)
    return integral
end

function makeHmatrix(f0::Array{Complex{Float64},2}, par::ParameterSet, z::Float64)
    Nx = size(f0,1)
    f0length = par.sim.subpixelpitch*Nx

    # Generates spatial frequency range at 256-bit precision to avoid inexact error
    truestep = BigFloat(one(BigFloat)/f0length)
    endpt = BigFloat(one(BigFloat)/(2*par.sim.subpixelpitch))
    spfreq = range(-endpt, stop=endpt-truestep, length=Nx)

    # Setup frequency axes
    fx = spfreq'
    fy = reverse(spfreq, dims=1)
    FXFY = fx.^2 .+ fy.^2

    # Fresnel propagation func. (see: Computational Fourier Optics p.54-55,63)
    h = exp(im*par.con.k0*z).*exp.((-im*pi*par.opt.lambda*z) .* FXFY)
    # Shifted for later fourier space calc and converted back to 64bit precision
    H = ComplexF64.(fftshift(h))
    return H
end

function calcPSF(imgspace::Space, objspace::Space, par::ParameterSet)
    pattern_stack = Array{Array{Complex{Float64},2},1}(undef, objspace.zlen)
    originimgs = complex(zeros(imgspace.xlen,imgspace.ylen,objspace.zlen))

    allocateStackMem(pattern_stack, objspace, imgspace, par)
    integratePSF(originimgs, pattern_stack, objspace, imgspace, par)

    #This is what happens when Shu Jia lab super resolution is applied
    if par.opt.a0 > 0.0
        steps::Int64 = 10
        stepz = par.opt.a0/steps
        Ha0 = makeHmatrix(originimgs[:,:,1], par, stepz)
        originimgs = itrfresnel!(originimgs, Ha0, steps, objspace)
    end

    return originimgs
end

function allocateStackMem(pattern_stack::Array{Array{Complex{Float64},2},1}, objspace::Space, imgspace::Space, par::ParameterSet)
    zmax = maximum(objspace.z)

    for p in 1:objspace.zlen
        IMGSIZE_REF_IL = cld((imgspace.xlen*abs(objspace.z[p])),zmax)
        halfWidth_IL =  max(IMGSIZE_REF_IL*par.sim.subvpix, 2*par.sim.subvpix)
        centerArea = max((imgspace.center - halfWidth_IL + 1) , 1):1:min((imgspace.center + halfWidth_IL - 1), imgspace.xlen)
        pattern_stack[p] = complex(zeros(Float64, length(centerArea), length(centerArea)))
    end

    return
end

function unfoldPattern(I1,pattern)
    middle = Int(cld(size(pattern,1),2))
    pattern[1:middle,1:middle] .= I1[:,:]
    pattern[1:middle,middle:end] .= reverse(I1, dims=2)
    pattern[middle:end, 1:end] .= reverse(pattern[1:middle,1:end], dims=1)
    return pattern
end

function integratePSF(originimgs::Array{Complex{Float64},3}, pattern_stack::Array{Array{Complex{Float64},2},1}, objspace::Space, imgspace::Space, par::ParameterSet)
    zmax = maximum(objspace.z)
    for j in 1:objspace.zlen

        IMGSIZE_REF_IL = cld((imgspace.xlen*abs(objspace.z[j])),zmax)
        halfWidth_IL =  max(IMGSIZE_REF_IL*par.sim.subvpix, 2*par.sim.subvpix)
        centerArea = Int.(max((imgspace.center - halfWidth_IL + 1) , 1):1:min((imgspace.center + halfWidth_IL - 1), imgspace.xlen))
        xL2length = length(centerArea[1]:imgspace.center)
        triangleIndices = falses(xL2length, xL2length)

        for X1 in 1:xL2length
            triangleIndices[1:X1,X1] .= true
        end

        xL2normsq = ((( imgspace.x[Int.(centerArea[1]:imgspace.center)]'.^2  .+
        imgspace.y[Int.(centerArea[1]:imgspace.center)].^2 ) .^0.5) ./ par.opt.M)

        v = xL2normsq.*(par.con.k*sin(par.con.alpha))
        u = 4*par.con.k*objspace.z[j]*(sin(par.con.alpha/2)^2)
        Koi = par.opt.M/((par.opt.fobj*par.opt.lambda)^2)*exp(-im*u/(4*(sin(par.con.alpha/2)^2)))

        I1 = complex(zeros(size(v)))
        I1[triangleIndices] = intpsf.(v[triangleIndices],u,par.opt.a0, par.con.alpha)
        I1[triangleIndices] .= I1[triangleIndices] .* Koi
        copyto!(I1,Symmetric(I1))

        pattern_stack[j] .= unfoldPattern(I1,pattern_stack[j])
        originimgs[centerArea,centerArea,j] .= pattern_stack[j][:,:]
    end
    return
end

function itrfresnel!(originimgs::Array{Complex{Float64},3}, Ha0::Array{Complex{Float64},2}, steps::Int64, objspace::Space)
    f0 = originimgs[:,:,1]
    p = plan_fft!(f0, [1,2])
    invp = inv(p)
    for h in 1:length(objspace.z)
        f0 .= originimgs[:,:,h]
        # Fourier space computation
        p*f0                        # Applies fft in place
        f0 .= f0.*(Ha0.^steps)      # Multiply by transfer function
                                    # each multiplication is an incremental proj
        invp\f0                     # Applies ifft in place
        originimgs[:,:,h] .= f0
    end
    return originimgs
end

end # psf module end
