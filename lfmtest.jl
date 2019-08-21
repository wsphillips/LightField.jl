using Revise
#################################
using LightField

(params, objectspace, mlaspace) = setup("configexample.toml")

imagespace = calcsize(params, objectspace)

psfstack = calcPSF(imagespace, objectspace, params)

mlarray = calcML(imagespace, mlaspace, params)
