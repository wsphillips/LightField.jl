module mla

using DSP
import LightField.params.ParameterSet, LightField.params.Space
export calcML
# TODO: Refactor this code.

function calcML(imgspace::Space, mlaspace::Space, par::ParameterSet)

    x1center = findfirst(imgspace.x .== 0)
    x2center = x1center
    x1MLdist = mlaspace.xlen
    x2MLdist = x1MLdist
    x1centerALL = vcat(x1center:-x1MLdist:1 , (x1center + x1MLdist):x1MLdist:imgspace.xlen)
    x1centerALL = sort!(x1centerALL)
    x2centerALL = x1centerALL
    patternML = complex(zeros(x1MLdist, x2MLdist))

    xL2norm = mlaspace.x.^2 .+ mlaspace.y'.^2
    patternML .= exp.(((-im*par.con.k)/(2*par.mla.fml)).*xL2norm);

    MLARRAY = complex(zeros(imgspace.xlen, imgspace.ylen))
    MLARRAY[x1centerALL, x2centerALL] .= 1

    #TODO: This is messy. Can be replaced with a filter function that auto crops
    # conv2() deprecated in favor or conv()
    MLARRAYfull = conv2(MLARRAY, patternML)
    border = fld((size(MLARRAYfull,1) - imgspace.xlen),2) + 1
    croprange = Int.(border:(imgspace.xlen + (border-1)))

    return MLARRAY = MLARRAYfull[croprange, croprange]
end

end # module
