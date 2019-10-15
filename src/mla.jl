module mla

using DSP
using ..params
export calcml

function calcml(img::Space, mlaspace::Space, par::ParameterSet)

    center = findfirst(img.x .== 0)
    mllen = mlaspace.xlen
    
    allcenters = vcat(center:-mllen:1, (center + mllen):mllen:img.xlen)
    sort!(allcenters) 

    a = complex(zeros(mllen,mllen))

    l2norm = mlaspace.x.^2 .+ mlaspace.y'.^2
    a .= exp.(((-im * par.con.k) / (2 * par.mla.fml)) .* l2norm);

    b = zeros(ComplexF64, img.xlen, img.ylen)
    b[allcenters, allcenters] .= 1

    #TODO: This is messy. Can be replaced with a filter function that auto crops
    # conv2() deprecated in favor or conv()
    c = conv(b, a)
    border = fld((size(c, 1) - img.xlen), 2) + 1
    crop = Int.(border:(img.xlen + (border-1)))

    return c[crop, crop]
end

end # module
