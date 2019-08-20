#%% USER-DEFINED PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd("/home/wsphil/Documents/LightField")

using SpecialFunctions
import DSP.conv2
using LinearAlgebra
using CuArrays.CUFFT
using FFTW
using FastGaussQuadrature
using PaddedViews
import Plots.gr; gr()
import Plots.heatmap
using JLD2, FileIO
using CUDAdrv
using CUDAnative
using CuArrays

function makeHmatrix(f0,dx,z,k0,lambda)
    Nx = size(f0,1)
    f0length = dx*Nx

    # Generates spatial frequency range at 256-bit precision to avoid inexact error
    truestep = BigFloat(one(BigFloat)/f0length)
    endpt = BigFloat(one(BigFloat)/(2*dx))
    spfreq = range(-endpt, stop=endpt-truestep, length=Nx)

    # Setup frequency axes
    fx = spfreq'
    fy = reverse(spfreq, dims=1)
    FXFY = fx.^2 .+ fy.^2

    # Fresnel propagation func. (see: Computational Fourier Optics p.54-55,63)
    h = exp(im*k0*z).*exp.((-im*pi*lambda*z) .* FXFY)
    # Shifted for later fourier space calc and converted back to 64bit precision
    H = ComplexF64.(fftshift(h))

    return H
end

function calcht(Himgs, Nnum, x3objspace,Xidx, Yidx, Zidx)
    Himgsize = size(Himgs,1)
    x3length = length(x3objspace)
    lenslets = cld(Himgsize,Nnum) # How many lenslets wide is the image
    if mod(lenslets,2) == 1 # Pad the image area
        imgsize = (lenslets+2)*Nnum
    else
        imgsize = (lenslets+3)*Nnum
    end
    Nnumpts = Nnum^2
    imcenter = cld(imgsize,2)
    imcenterinit = imcenter - cld(Nnum,2)
    x1 = Xidx[Zidx.==1]
    y1 = Yidx[Zidx.==1]
    xpoints = imcenterinit .+ x1
    ypoints = imcenterinit .+ y1
    shiftx = cld(Nnum,2) .- xpoints
    shifty = cld(Nnum,2) .- ypoints
    cropstart = Int(imcenter - (Himgsize-1)/2)
    cropstop = Int(imcenter + (Himgsize-1)/2)

    projection = zeros(imgsize, imgsize, Nnumpts)
    for i in 1:Nnumpts
        projection[xpoints[i],ypoints[i],i] = 1.0
    end
    Ht = zeros(size(Himgs))
    Hkernel = reverse(Himgs, dims=1)
    Hkernel .= reverse(Hkernel, dims=2)

    ### similar to reconstruction
    for i in 1:Nnumpts
        projection_GPU = (projection[:,:,i])
        Backprojection = (zeros(size(projection, 1), size(projection, 1), x3length ))

        for j in 1:Nnum
            for k in 1:Nnum
                for p in 1:x3length

                indexes = (Xidx .== j) .& (Yidx .== k) .& (Zidx .== p)
                Hkernel_GPU = (Himgs[:,:,indexes])

                Hkernel_GPU = convolve2(projection_GPU,Hkernel_GPU, AF_CONV_DEFAULT, AF_CONV_AUTO)


                 Backprojection[j:Nnum:end, k:Nnum:end, p] = Hkernel_GPU[j:Nnum:end, k:Nnum:end]

                end
            end
        end
        println("innerloop done")

        Backprojection = Backprojection[cropstart:cropstop, cropstart:cropstop, :]
        Backprojection = translate(Backprojection, shiftx[i], shifty[i], size(Backprojection,1), size(Backprojection,2))

        #back to host here
        indexes2 = (Xidx .== x1[i]) .& (Yidx .== y1[i])
        Ht[:,:,indexes2] = Array(Backprojection)

        println("full loop")
    end
    Backprojection = finalize(Backprojection)
    convolution = finalize(convolution)
    Hkernel_GPU = finalize(Hkernel_GPU)
    afgc(1000)

    ### end similar to reconstruction

    return Ht
end


"These are scaling terms to convert array subscripts to obj space"
XREF = cld(length(x1objspace),2)
YREF = cld(length(x1objspace),2)
Xidx = repeat(1:length(x1objspace), outer=(length(x2objspace)*length(x3objspace)))
Yidx = repeat(1:length(x2objspace), inner=(length(x1objspace)*length(x3objspace)))
Zidx = repeat(1:length(x3objspace), inner=(length(x1objspace)*length(x2objspace)))

"For shifting the images relative to object space"
SHIFTX = Float32.((Xidx.-XREF).*OSR)
SHIFTY = Float32.((Yidx.-YREF).*OSR)

"Resampling points -- just to ensure the center point remains consistent"
half1 = centerPT:-OSR:1
half2 = (centerPT + OSR):OSR:x1_imgspacelength
samples = vcat(half1,half2)
sort!(samples)
"Make lowpass 2D sinc filter for resampling"
x1 = range(-2, stop=2, length=13) #NOTE THIS ASSUMES OSR OF 3!!!
x2 = x1'
sinc2d(x,y) = (sinc(x)*sinc(y))

# Setup for GPU compute--conservative memory usage atm...
afgc(1000)        #garbage collect GPU memory to be sure its cleared
imgsperlayer = Nnum*Nnum
H = makeHmatrix(psfWAVE_STACK[:,:,1],subpixelpitch,d,k0,lambda)
Hlayer = complex(zeros(size(MLARRAY,1),size(MLARRAY,2),imgsperlayer))
Himgs = zeros(length(samples),length(samples), Nnum*Nnum*length(x3objspace))
sincfilter = sinc2d.(x1,x2).*0.330

"Prep kernels onto GPU memory"
MLARRAYGPU = AFArray(MLARRAY)
HGPU = AFArray(H)
sincfilterGPU = AFArray(sincfilter)
end #prep timer end
#NOTE: Amend with calls to afgc(1000) !!!!
println("Computing LF PSFs (2/3)")
@time begin
for layer in 1:length(x3objspace)
    stackstart = 1 + (layer-1)*imgsperlayer
    stackend   = layer*imgsperlayer
    baseimg = AFArray(psfWAVE_STACK[:,:,layer])
    for img in 1:imgsperlayer
        stackbin = translate(baseimg, SHIFTX[img], SHIFTY[img], size(MLARRAY,1), size(MLARRAY,2), AF_INTERP_NEAREST)
        @afgc stackbin = stackbin .* MLARRAYGPU
        #Fresnel 2d one step size of b0
        fft2!(stackbin)
        @afgc stackbin = stackbin .* HGPU
        ifft2!(stackbin)

        stackbin = translate(stackbin, -SHIFTX[img], -SHIFTY[img], size(MLARRAY,1), size(MLARRAY,2), AF_INTERP_NEAREST)
        @afgc stackbin = abs2.(stackbin)

        #Non original code: addition of low pass filter before downsampling
        stackbin = convolve2(stackbin, sincfilterGPU, AF_CONV_DEFAULT, AF_CONV_AUTO)
        Hlayer[:,:,img] = Array(stackbin)
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

#Scratch for viewing images as tiff in ImageJ
#afgc(1000)
#Ht = calcht(Himgs, Nnum, x3objspace,Xidx, Yidx, Zidx)


@save "fullSimulation.jld2" Himgs Himgs32bit M Nnum SHIFTX SHIFTY Xidx Yidx Zidx pixelPitch x3objspace
#occupied = Himgs32bit[:,1,:] .!= 0


#skimageio.imsave("Himgs32bit.tiff", Himgs32bit)
#A = skimageio.imread("foo.tiff"
