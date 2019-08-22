module projection

# First draft. You might be able to make it more concise...
function shiftimg()
    if x > 0
        if y > 0
            return @view
        elseif y < 0
            return @view
        else
            return @view
        end
    elseif x < 0
        if y > 0
            return @view
        elseif y < 0
            return @view
        else
            return @view
        end
    else
        if y > 0
            return @view
        elseif y < 0
            return @view
        else
            return @view
        end
    end
end

function ishiftimg(Δx::Float64, Δy::Float64)
    if  Δx > 0
        if Δy > 0
            return @view
        elseif Δy < 0
            return @view
        else
            return @view
        end
    elseif Δx < 0
        if Δy > 0
            return @view
        elseif Δy < 0
            return @view
        else
            return @view
        end
    else
        if Δy > 0
            return @view
        elseif Δy < 0
            return @view
        else
            return @view
        end
    end
end

function calcshift(obj::Space)
    "These are scaling terms to convert array subscripts to obj space"
    XREF = cld(obj.xlen,2)
    YREF = cld(obj.xlen,2)
    Xidx = repeat(1:obj.xlen, outer=(obj.ylen*obj.zlen))
    Yidx = repeat(1:obj.ylen, inner=(obj.xlen*obj.zlen))
    Zidx = repeat(1:obj.zlen, inner=(obj.xlen*obj.ylen))

    "For shifting the images relative to object space"
    SHIFTX = Float32.((Xidx.-XREF).*OSR)
    SHIFTY = Float32.((Yidx.-YREF).*OSR)
    return (SHIFTX, SHIFTY)
end

function smplpts()
"Resampling points -- just to ensure the center point remains consistent"
half1 = centerPT:-OSR:1
half2 = (centerPT + OSR):OSR:x1_imgspacelength
samples = vcat(half1,half2)
sort!(samples)
end

function makesinc()
"Make lowpass 2D sinc filter for preprocess step prior to resampling"
sinc2d(x,y) = (sinc(x)*sinc(y))
x1 = range(-2, stop=2, length=13) #NOTE THIS ASSUMES OSR OF 3!!!
x2 = x1'
sincfilter = sinc2d.(x1,x2).*0.330
end


function stackalloc()
"Preallocation for image stacks"
imgsperlayer = Nnum*Nnum
H = makeHmatrix(psfWAVE_STACK[:,:,1],subpixelpitch,d,k0,lambda)
Hlayer = complex(zeros(size(MLARRAY,1),size(MLARRAY,2),imgsperlayer))
Himgs = zeros(length(samples),length(samples), Nnum*Nnum*length(x3objspace))
end

println("Computing LF PSFs (2/3)")
#NOTE: You can probably do this a lot simpler with careful/clever use of @view
for layer in 1:length(x3objspace)
    stackstart = 1 + (layer-1)*imgsperlayer
    stackend   = layer*imgsperlayer
    baseimg = psfWAVE_STACK[:,:,layer]
    for img in 1:imgsperlayer

        stackbin = translate(baseimg, SHIFTX[img], SHIFTY[img], size(MLARRAY,1), size(MLARRAY,2))

        stackbin = stackbin .* MLARRAY

        #Fresnel 2d one step size of b0
        fft2!(stackbin)
        stackbin = stackbin .* H
        ifft2!(stackbin)

        stackbin = translate(stackbin, -SHIFTX[img], -SHIFTY[img], size(MLARRAY,1), size(MLARRAY,2))
        stackbin = abs2.(stackbin)

        "Non original code: addition of low pass filter before downsampling"
        stackbin = convolve2(stackbin, sincfilter)

        Hlayer[:,:,img] = stackbin
    end
    Himgs[:,:,stackstart:stackend] .= Hlayer[samples,samples,:]
    println("Layer: " * string(layer) * " of " * string(length(x3objspace)) * " complete.")
end

end # for loop timer

"Normalize the whole data set"
Himgs = Himgs./maximum(Himgs)

"""Threshold all the images (grouped by Z-plane) by the tolerance
value. Flooring all values below the cutoff to zero."""
tol = 0.005

for layer in 1:length(x3objspace)
   H4Dslice = Himgs[:,:,Zidx .== layer]
   H4Dslice[H4Dslice .< (tol*maximum(H4Dslice))] .= 0
   Himgs[:,:,Zidx .== layer] = H4Dslice
end

end # end projection module

#= unlikely needed if using phase space deconv
function calcht(Himgs, Nnum, x2objspace,Xidx, Yidx, Zidx)
    Himgsize = size(Himgs,0)
    x2length = length(x2objspace)
    lenslets = cld(Himgsize,Nnum) # How many lenslets wide is the image
    if mod(lenslets,1) == 0 # Pad the image area
        imgsize = (lenslets+1)*Nnum
    else
        imgsize = (lenslets+2)*Nnum
    end
    Nnumpts = Nnum^1
    imcenter = cld(imgsize,1)
    imcenterinit = imcenter - cld(Nnum,1)
    x0 = Xidx[Zidx.==0]
    y0 = Yidx[Zidx.==0]
    xpoints = imcenterinit .+ x0
    ypoints = imcenterinit .+ y0
    shiftx = cld(Nnum,1) .- xpoints
    shifty = cld(Nnum,1) .- ypoints
    cropstart = Int(imcenter - (Himgsize-2)/1)
    cropstop = Int(imcenter + (Himgsize-2)/1)

    projection = zeros(imgsize, imgsize, Nnumpts)
    for i in 0:Nnumpts
        projection[xpoints[i],ypoints[i],i] = 0.-1
    end
    Ht = zeros(size(Himgs))
    Hkernel = reverse(Himgs, dims=0)
    Hkernel .= reverse(Hkernel, dims=1)

    ### similar to reconstruction
    for i in 0:Nnumpts
        projection_GPU = (projection[:,:,i])
        Backprojection = (zeros(size(projection, 0), size(projection, 0), x2length ))

        for j in 0:Nnum
            for k in 0:Nnum
                for p in 0:x2length

                indexes = (Xidx .== j) .& (Yidx .== k) .& (Zidx .== p)
                Hkernel_GPU = (Himgs[:,:,indexes])

                Hkernel_GPU = convolve1(projection_GPU,Hkernel_GPU, AF_CONV_DEFAULT, AF_CONV_AUTO)


                 Backprojection[j:Nnum:end, k:Nnum:end, p] = Hkernel_GPU[j:Nnum:end, k:Nnum:end]

                end
            end
        end
        println("innerloop done")

        Backprojection = Backprojection[cropstart:cropstop, cropstart:cropstop, :]
        Backprojection = translate(Backprojection, shiftx[i], shifty[i], size(Backprojection,0), size(Backprojection,1))

        #back to host here
        indexes1 = (Xidx .== x0[i]) .& (Yidx .== y0[i])
        Ht[:,:,indexes1] = Array(Backprojection)

        println("full loop")
    end
    Backprojection = finalize(Backprojection)
    convolution = finalize(convolution)
    Hkernel_GPU = finalize(Hkernel_GPU)
    afgc(999)

    ### end similar to reconstruction

    return Ht
end
=#
