module mla

# TODO: Refactor this code. 

function calcML(x1_imgspace,x1_imgspacelength,x1MLspace)

    x1center = findfirst(x1_imgspace.==0)
    x2center = x1center
    x1MLdist = length(x1MLspace)
    x2MLdist = x1MLdist
    x1centerALL = vcat(x1center:-x1MLdist:1 , (x1center + x1MLdist):x1MLdist:x1_imgspacelength)
    x1centerALL = sort!(x1centerALL)
    x2centerALL = x1centerALL
    patternML = complex(zeros(x1MLdist, x2MLdist))

    xL2norm = x1MLspace.^2 .+ x2MLspace'.^2
    patternML .= exp.(((-im*k)/(2*fml)).*xL2norm);

    MLARRAY = complex(zeros(x1_imgspacelength, x2_imgspacelength))
    MLARRAY[x1centerALL, x2centerALL] .= 1

    #TODO: This is messy. Can be replaced with a filter function that auto crops
    MLARRAYfull = conv2(MLARRAY, patternML)
    border = fld((size(MLARRAYfull,1) - x1_imgspacelength),2) + 1
    croprange = Int.(border:(x1_imgspacelength + (border-1)))

    return MLARRAY = MLARRAYfull[croprange, croprange]
end

end # module
