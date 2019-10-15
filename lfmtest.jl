using Revise
#################################
using LightField

cd("/home/wikphi@ad.cmm.se/LightField.jl/")

lf = setup("configexample.toml")
imagespace = calcsize(lf)
mlarray = calcML(imagespace, mlaspace, params)

psfstack = originPSFproj(lf)
Himgs = propagate(psfstack, mlarray, mlaspace, imagespace, objectspace, params)



