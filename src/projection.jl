module projection

using FFTW


# First draft. You might be able to make it more concise...
function shiftimg(img::Union{Array{Complex{Float64},2},Array{Float64,2}}, Δx::Int64, Δy::Int64)
    if  Δx > 0
        if Δy > 0
            return PaddedView(0, img[1+Δy:end, 1:end-Δx], size(img), (1, 1+Δx))
        elseif Δy < 0
            return PaddedView(0, img[1:end+Δy, 1:end-Δx], size(img), (1-Δy, 1+Δx))
        else
            return PaddedView(0, img[:, 1:end-Δx], size(img), (1, 1+Δx))
        end
    elseif Δx < 0
        if Δy > 0
            return PaddedView(0, img[1+Δy:end, 1-Δx:end], size(img))
        elseif Δy < 0
            return PaddedView(0, img[1:end+Δy, 1-Δx:end], size(img), (1-Δy, 1))
        else
            return PaddedView(0, img[:, 1-Δx:end], size(img))
        end
    else
        if Δy > 0
            return PaddedView(0, img[1+Δy:end, :], size(img))
        elseif Δy < 0
            return PaddedView(0, img[1:end+Δy, :], size(img), (1-Δy, 1))
        else
            return PaddedView(0, img[:,:], size(img))
        end
    end
end

function calcshifts(obj::Space, par::ParameterSet)
    "These are scaling terms to convert array subscripts to obj space"
    XREF = obj.center
    YREF = XREF
    Xidx = repeat(1:obj.xlen, outer=(obj.ylen*obj.zlen))
    Yidx = repeat(1:obj.ylen, inner=(obj.xlen*obj.zlen))
    Zidx = repeat(1:obj.zlen, inner=(obj.xlen*obj.ylen))

    "For shifting the images relative to object space"
    SHIFTX = Int64.((Xidx.-XREF).*par.sim.osr)
    SHIFTY = Int64.((Yidx.-YREF).*par.sim.osr)
    return (SHIFTX, SHIFTY, Zidx)
end

"Generates sampling points -- just to ensure the center point remains consistent"
function sample(img::Space, par::ParameterSet)
    half1 = img.center:-par.sim.osr:1
    half2 = (img.center + par.sim.osr):par.sim.osr:img.xlen
    samples = vcat(half1,half2)
    sort!(samples)
    return samples
end
"Make lowpass 2D sinc filter for preprocess step prior to resampling"
function make2dsinc()
    sinc2d(x,y) = (sinc(x)*sinc(y))
    x1 = range(-2, stop=2, length=13) #NOTE THIS ASSUMES OSR OF 3!!!
    x2 = x1'
    return sinc2d.(x1,x2).*0.330
end

"Preallocation for image stacks"
function initproject(originpsf::Array{Complex{Float64},3}, mla::Space, obj::Space, par::ParameterSet)
    imgsperlayer = par.sim.vpix^2

    H = makeHmatrix(originpsf[:,:,1],par,par.opt.d)
    Himgs = zeros(mla.xlen,mla.ylen, imgsperlayer*objspace.zlen)
    Himgtemp = zeros(Complex{Float64}, size(Himgs,1), size(Himgs,1), imgsperlayer)

    FFTW.set_num_threads(40)
    plan = plan_fft!(Htemp,[1,2], flags=FFTW.MEASURE)

    return (imgsperlayer, H, Himgtemp, Himgs, plan)
end

function project!(originpsf::Array{Complex{Float64},3}, mlarray::Array{Complex{Float64},2}, mla::Space, obj::Space, par::ParameterSet)

    (SHIFTX, SHIFTY, Zidx) = calcshifts(obj, par)
    (imgsperlayer, H, Himgtemp, Himgs, plan) = initproject(originpsf, mla, obj, par)

    for layer in 1:obj.zlen
        stackstart = 1 + (layer-1)*imgsperlayer
        stackend   = layer*imgsperlayer
        @views Himgtemp  .= shift.(psforigin[:,:,layer], SHIFTX, SHIFTY) .* mlarray
        plan*Himgtemp
        @views Himgtemp .= Himgtemp .* H
        plan\Himgtemp
        @views Himgs[:,:,stackstart:stackend] .= abs2.(shift.(Himgtemp,  -SHIFTX, -SHIFTY))
    end
    
    return Himgs
end

function postprocess(Himgs::Array{Complex{Float64},3}, objspace::Space, Zidx::Array{Int64,1})

    #=TODO: Needs to be revised for new pipeline
    "Non original code: addition of low pass filter before downsampling"
    stackbin = conv(stackbin, sincfilter)[croppedrng, croppedrng]
    =#
    "Normalize the whole data set"
    Himgs = Himgs./maximum(Himgs)

    """Threshold all the images (grouped by Z-plane) by the tolerance
    value. Flooring all values below the cutoff to zero."""
    tol = 0.005

    for layer in 1:objspace.zlen
        H4Dslice = Himgs[:,:,Zidx .== layer]
        H4Dslice[H4Dslice .< (tol*maximum(H4Dslice))] .= 0
        Himgs[:,:,Zidx .== layer] = H4Dslice
    end
    return Himgs
end

end # end projection module
