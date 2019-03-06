module psfsize

function calcsize()

end

function psfline(x1testspace, M, k, alpha, x3objmax, fobj)
    xL2normsq = abs.(x1testspace)./M
    v = xL2normsq.*(k*sin(alpha))
    u = 4*k*x3objmax*(sin(alpha/2)^2)
    Koi = M/((fobj*lambda)^2)*exp(-im*u/(4*(sin(alpha/2)^2)))
    psfLine = complex(zeros(length(v)))

    psfLine = intfxn(nodes,weights,0,alpha,v,u)
    psfLine = psfLine.*Koi
    psfLine = Float64.(abs.(psfLine.^2)./(maximum(abs.(psfLine.^2))))
end

function makeimgspace()

end
