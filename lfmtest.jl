using Revise
#################################
using LightField

cd("/home/wikphi@ad.cmm.se/LightField.jl/")
@time begin
(params, objectspace, mlaspace) = setup("configexample.toml")
imagespace = calcsize(params, objectspace)
psfstack = originPSFproj(imagespace, objectspace, params)
mlarray = calcML(imagespace, mlaspace, params)
Himgs = propagate(psfstack, mlarray, mlaspace, imagespace, objectspace, params)
end #timer
