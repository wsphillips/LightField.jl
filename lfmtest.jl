using Revise
#################################
using LightField

cd("/home/wikphi@ad.cmm.se/LightField.jl/")

lf = setup("configexample.toml")
imagespace = calcsize(lf)
mlarray = calcML(imagespace, mlaspace, params)

psfstack = originPSFproj(lf)
Himgs = propagate(psfstack, mlarray, mlaspace, imagespace, objectspace, params)



#= Scratch space below =#


function foo()
   s::Vector{Int} = round.(rand(62).*100)
    sort!(s)
    # TODO: fix scope shit here
    while mod(length(s), 7) > 0
        s = s[2:end-1]
    end
    return s
end

test = foo()

