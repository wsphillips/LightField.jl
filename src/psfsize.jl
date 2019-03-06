module psfsize

function calcsize()

end

function psfline()

end

function makeimgspace()

end

function intchange(x::Float64,a::Float64,b::Float64)
    newx = (((b-a)/2)*x) + ((a + b)/2)
    return newx
end

function calcPSFFT(x1testspace, M, k, alpha, x3objmax, fobj)

    (nodes, weights) = gausslegendre(500)



    function psf(theta,v,u)
        if a0 > 0.0
            intensity = (sqrt(cos(theta)).*besselj.(0,((v.*sin(theta))./sin(alpha))).*(exp((im*u*sin(theta/2)^2)/(2*sin(alpha/2)^2))*sin(theta)))
        else
            intensity = (sqrt(cos(theta)).*besselj.(0,((v.*sin(theta))./sin(alpha))).*(exp((-im*u*sin(theta/2)^2)/(2*sin(alpha/2)^2))*sin(theta)))
        end
        return intensity
    end

    function intfxn(nodes,weights, a, b, v, u)
        integral = ((b-a)/2) .* sum(psf.(intchange.(nodes', a, b),v,u).*weights', dims=2)
        return integral
    end

    xL2normsq = abs.(x1testspace)./M
    v = xL2normsq.*(k*sin(alpha))
    u = 4*k*x3objmax*(sin(alpha/2)^2)
    Koi = M/((fobj*lambda)^2)*exp(-im*u/(4*(sin(alpha/2)^2)))
    psfLine = complex(zeros(length(v)))

    psfLine = intfxn(nodes,weights,0,alpha,v,u)
    psfLine = psfLine.*Koi
    psfLine = Float64.(abs.(psfLine.^2)./(maximum(abs.(psfLine.^2))))

    return psfLine
end
