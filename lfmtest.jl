using Revise
#################################
using LightField

cd("/home/wikphi@ad.cmm.se/LightField.jl/")

(params, objectspace, mlaspace) = setup("configexample.toml")
imagespace = calcsize(params, objectspace)
psfstack = originPSFproj(imagespace, objectspace, params)
mlarray = calcML(imagespace, mlaspace, params)

Himgs = propagate(psfstack, mlarray, mlaspace, objectspace, params)
