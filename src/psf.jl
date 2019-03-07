module psf

import SpecialFunctions.besselj
import FastGaussQuadrature.gausslegendre

export intpsf

function intervalchange(x::Float64,a::Float64,b::Float64)

    newx = x*(b-a)/2 + (a+b)/2

    return newx
end

function psf(θ::Float64, α::Float64, v::Float64, u::Float64, a₀::Float64)
    if a₀ > 0.0
        # PSF for classic LFM
        I = sqrt(cos(θ)) * besselj(0, v*sin(θ)/sin(α)) * exp((im*u*sin(θ/2)^2)/
                                                        (2*sin(α/2)^2)) * sin(θ)
    else
        # PSF as given by Li et al. 2018. Used when MLA is offset from NIP by
        # distance a₀
        I = sqrt(cos(θ)) * besselj(0,v*sin(θ)/sin(α)) * exp((-im*u*sin(θ/2)^2)/
                                                        (2*sin(α/2)^2)) * sin(θ)
    end
    return I
end

function intpsf(v::Float64, u::Float64, a₀::Float64, α::Float64)

    # NOTE: Number of quadrature nodes can be tuned for speed or accuracy
    (x, weights) = gausslegendre(100)

    θ = intervalchange.(x, 0.0, α)

    integral = (α/2) * sum(psf.(θ, α, v, u, a₀) .* weights)

    return integral
end

end # module
