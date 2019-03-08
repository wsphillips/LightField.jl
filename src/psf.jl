module psf

import SpecialFunctions.besselj
import FastGaussQuadrature.gausslegendre

export intpsf

function intervalchange(x::Float64,a::Float64,b::Float64)

    newx = x*(b-a)/2 + (a+b)/2

    return newx
end

function lfmpsf(θ::Float64, α::Float64, v::Float64, u::Float64, a₀::Float64)

    if a₀ > 0.0
        # PSF for classic LFM
        I = sqrt(cos(θ)) * besselj(0, v*sin(θ)/sin(α)) * exp((im*u*sin(θ/2)^2)/
                                                                (2*sin(α/2)^2))
                                                                    * sin(θ)
    else
        # PSF as given by Li et al. 2018. Used when MLA is offset from NIP by
        # distance a₀
        I = sqrt(cos(θ)) * besselj(0,v*sin(θ)/sin(α)) * exp((-im*u*sin(θ/2)^2)/
                                                                (2*sin(α/2)^2))
                                                                    * sin(θ)
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

# TODO: Refactor this code. Check for package add to do symmetric matrix

function calcPSF(x3objspace,k,M,fobj,lambda,alpha,IMGSIZE_REF,subNnum,centerPT,x1_imgspacelength,x2_imgspacelength, x3objmax, dx, k0, a0)

    pattern_stack = Array{Array{Complex{Float64},2},1}(undef, length(x3objspace))

    originimgs = complex(zeros(x1_imgspacelength,x1_imgspacelength,length(x3objspace)))


    allocateStackMem(pattern_stack, x3objspace, centerPT)

    integratePSF(originimgs, pattern_stack, x3objspace, centerPT, fobj, lambda, alpha, M, dx, k0, k, a0)

    #This is what happens when Shu Jia lab super resolution is applied
    if a0 > 0.0
        steps = 10
        stepz = a0/10
        Ha0 = makeHmatrix(originimgs[:,:,1],subpixelpitch,stepz,k0,lambda)
        originimgs = itrfresnel_GPU!(originimgs, Ha0, steps)
    end

    return originimgs
end

function allocateStackMem(pattern_stack, x3objspace, centerPT)
    for p in 1:length(x3objspace)
        IMGSIZE_REF_IL = cld((IMGSIZE_REF*abs(x3objspace[p])),x3objmax)

        halfWidth_IL =  max(IMGSIZE_REF_IL*subNnum, 2*subNnum)

        centerArea = max((centerPT - halfWidth_IL + 1) , 1):1:min((centerPT + halfWidth_IL - 1) , x1_imgspacelength)

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

function itrfresnel_GPU!(originimgs, Ha0, steps)

    Ha0_GPU = cu(Ha0)

    originimgs_GPU = cu(originimgs)

    f0 = cu(originimgs_GPU[:,:,1])

    p = plan_fft!(f0, [1,2])

    invp = inv(p)

    for h in 1:length(x3objspace)

        f0 .= originimgs_GPU[:,:,h]

        # Fourier space computation

        p*f0               # Applies fft in place

        f0 .= f0.*(Ha0_GPU.^steps)        # Multiply by transfer function
                             # each multiplication is an incremental proj
        invp\f0              # Applies ifft in place

        originimgs_GPU[:,:,h] .= f0

    end

    originimgs = collect(originimgs_GPU)

    Ha0_GPU = finalize(Ha0_GPU)

    f0 = finalize(f0)

    return originimgs
end

function integratePSF(originimgs, pattern_stack, x3objspace, centerPT, fobj, lambda, alpha, M, dx, k0, k, a0)

    for j in 1:length(x3objspace)

        IMGSIZE_REF_IL = cld((IMGSIZE_REF*abs(x3objspace[j])),x3objmax)

        halfWidth_IL =  max(IMGSIZE_REF_IL*subNnum, 2*subNnum)

        centerArea = Int.(max((centerPT - halfWidth_IL + 1) , 1):1:min((centerPT + halfWidth_IL - 1) , x1_imgspacelength))

        xL2length = length(centerArea[1]:centerPT)

        triangleIndices = falses(xL2length, xL2length)

        for X1 in 1:xL2length

            triangleIndices[1:X1,X1] .= true

        end

        xL2normsq = ((( x1_imgspace[centerArea[1]:centerPT]'.^2  .+
        x2_imgspace[centerArea[1]:centerPT].^2 ) .^0.5) ./M)

        v = xL2normsq.*(k*sin(alpha))

        u = 4*k*x3objspace[j]*(sin(alpha/2)^2)

        Koi = M/((fobj*lambda)^2)*exp(-im*u/(4*(sin(alpha/2)^2)))

        I1 = complex(zeros(size(v)))

        I1[triangleIndices] = intfxn(nodes,weights,0,alpha,v[triangleIndices],u)

        I1[triangleIndices] .= I1[triangleIndices] .* Koi

        copyto!(I1,Symmetric(I1))

        pattern_stack[j] .= unfoldPattern(I1,pattern_stack[j])

        originimgs[centerArea,centerArea,j] .= pattern_stack[j][:,:]
    end
    return
end

end # module
