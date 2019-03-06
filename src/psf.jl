module psf

import SpecialFunctions.besselj
import FastGaussQuadrature.gausslegendre

export intpsf

function intervalchange(x::Float64,a::Float64,b::Float64)
    newx = (((b-a)/2.0)*x) + ((a + b)/2.0)
    return newx
end

function psf(θ::Float64, α::Float64, v, u, a₀::Float64)
    if a₀ > 0.0
        intensity = sqrt(cos(θ)) * besselj(0, v*sin(θ)/sin(α)) * exp((im*u*sin(θ/2)^2)/(2*sin(α/2)^2)) * sin(θ)
    else
        intensity = (sqrt(cos(theta)).*besselj.(0,((v.*sin(theta))./sin(alpha))).*(exp((-im*u*sin(theta/2)^2)/(2*sin(alpha/2)^2))*sin(theta)))
    end
    return intensity
end

function intpsf(a::Float64, b::Float64, v, u, a0::Float64)
    (nodes, weights) = gausslegendre(500)
    integral = ((b-a)/2) .* sum(psf.(intervalchange.(nodes, a, b),v,u).*weights, dims=2)
    return integral
end

end
