module projection

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

function smplpts(img::Space, par::ParameterSet)
    "Resampling points -- just to ensure the center point remains consistent"
    half1 = img.center:-par.sim.osr:1
    half2 = (img.center + par.sim.osr):par.sim.osr:img.xlen
    samples = vcat(half1,half2)
    sort!(samples)
    return samples
end

function makesinc()
    "Make lowpass 2D sinc filter for preprocess step prior to resampling"
    sinc2d(x,y) = (sinc(x)*sinc(y))
    x1 = range(-2, stop=2, length=13) #NOTE THIS ASSUMES OSR OF 3!!!
    x2 = x1'
    return sinc2d.(x1,x2).*0.330
end

function stackalloc()
"Preallocation for image stacks"
imgsperlayer = Nnum*Nnum
H = makeHmatrix(psfWAVE_STACK[:,:,1],subpixelpitch,d,k0,lambda)
Hlayer = complex(zeros(size(MLARRAY,1),size(MLARRAY,2),imgsperlayer))
Himgs = zeros(length(samples),length(samples), Nnum*Nnum*objspace.zlen)
end
println("Computing LF PSFs (2/3)")
#NOTE: You can probably do this a lot simpler with careful/clever use of @view
#       I started to hash out the idea below. Verify. Probably need if-else for
# each shift case (positive, negative and zero). Could be a separate function that returns a view or range.
for layer in 1:objspace.zlen
    stackstart = 1 + (layer-1)*imgsperlayer
    stackend   = layer*imgsperlayer
    for img in 1:imgsperlayer

        @views stackbin .= shift(psforigin[:,:,layer], SHIFTX, SHIFTY) .* MLARRAY

        #Fresnel 2d one step size of b0
        fft2!(stackbin)
        stackbin = stackbin .* H
        ifft2!(stackbin)

        @views finalbin .= abs2.(shift(stackbin, -SHIFTX, -SHIFTY))

        "Non original code: addition of low pass filter before downsampling"
        stackbin = conv(stackbin, sincfilter)[croppedrng, croppedrng]

        Hlayer[:,:,img] = stackbin
    end
    Himgs[:,:,stackstart:stackend] .= Hlayer[samples,samples,:]
end

end # for loop timer

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

end # end projection module
