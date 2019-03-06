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

function calcPSF(x3objspace,k,M,fobj,lambda,alpha,IMGSIZE_REF,subNnum,centerPT,x1_imgspacelength,x2_imgspacelength, x3objmax, dx, k0, a0)

    function allocateStackMem(pattern_stack, x3objspace, centerPT)
        for p in 1:length(x3objspace)
            IMGSIZE_REF_IL = cld((IMGSIZE_REF*abs(x3objspace[p])),x3objmax)
            halfWidth_IL =  max(IMGSIZE_REF_IL*subNnum, 2*subNnum)
            centerArea = max((centerPT - halfWidth_IL + 1) , 1):1:min((centerPT + halfWidth_IL - 1) , x1_imgspacelength)
            pattern_stack[p] = complex(zeros(Float64, length(centerArea), length(centerArea)))
        end
        return
    end

    function unfoldPattern(I1,pattern)
        middle = Int(cld(size(pattern,1),2))
        pattern[1:middle,1:middle] .= I1[:,:]
        pattern[1:middle,middle:end] .= reverse(I1, dims=2)
        pattern[middle:end, 1:end] .= reverse(pattern[1:middle,1:end], dims=1)
        return pattern
    end

    function itrfresnel_GPU!(originimgs, Ha0, steps)
        Ha0_GPU = cu(Ha0)
        originimgs_GPU = cu(originimgs)
        f0 = cu(originimgs_GPU[:,:,1])
        p = plan_fft!(f0, [1,2])
        invp = inv(p)
        for h in 1:length(x3objspace)
            f0 .= originimgs_GPU[:,:,h]
            # Fourier space computation
            p*f0               # Applies fft in place

            f0 .= f0.*(Ha0_GPU.^steps)        # Multiply by transfer function
                                 # each multiplication is an incremental proj

            invp\f0              # Applies ifft in place

            originimgs_GPU[:,:,h] .= f0

        end
        originimgs = collect(originimgs_GPU)
        Ha0_GPU = finalize(Ha0_GPU)
        f0 = finalize(f0)
        return originimgs
    end

    function integratePSF(originimgs, pattern_stack, x3objspace, centerPT, fobj, lambda, alpha, M, dx, k0, k, a0)

        (nodes, weights) = gausslegendre(500)

        function intchange(x,a,b)
            y1 = (((b-a)/2)*x) + ((a + b)/2)
            return y1
        end

        function psf(theta,v,u)
            if a0 > 0.0
                 intensity = (sqrt(cos(theta)).*besselj.(0,((v.*sin(theta))/sin(alpha))).*(exp((im*u*sin(theta/2)^2)/(2*sin(alpha/2)^2))*sin(theta)))
            else
                 intensity = (sqrt(cos(theta)).*besselj.(0,((v.*sin(theta))/sin(alpha))).*(exp((-im*u*sin(theta/2)^2)/(2*sin(alpha/2)^2))*sin(theta)))
            end
            return intensity
        end

        function intfxn(nodes,weights, a, b, v, u)
            integral = ((b-a)/2) .* sum(psf.(intchange.(nodes', a, b),v,u).*weights', dims=2)
            return integral
        end

        for j in 1:length(x3objspace)

            IMGSIZE_REF_IL = cld((IMGSIZE_REF*abs(x3objspace[j])),x3objmax)
            halfWidth_IL =  max(IMGSIZE_REF_IL*subNnum, 2*subNnum)
            centerArea = Int.(max((centerPT - halfWidth_IL + 1) , 1):1:min((centerPT + halfWidth_IL - 1) , x1_imgspacelength))

            println("size of center area = " * string(length(centerArea)) * "X" * string(length(centerArea)))

            xL2length = length(centerArea[1]:centerPT)
            triangleIndices = falses(xL2length, xL2length)
            for X1 in 1:xL2length
                triangleIndices[1:X1,X1] .= true
            end
            xL2normsq = ((( x1_imgspace[centerArea[1]:centerPT]'.^2  .+  x2_imgspace[centerArea[1]:centerPT].^2 ) .^0.5) ./M)
            v = xL2normsq.*(k*sin(alpha))
            u = 4*k*x3objspace[j]*(sin(alpha/2)^2)
            Koi = M/((fobj*lambda)^2)*exp(-im*u/(4*(sin(alpha/2)^2)))
            I1 = complex(zeros(size(v)))


            I1[triangleIndices] = intfxn(nodes,weights,0,alpha,v[triangleIndices],u)

            I1[triangleIndices] .= I1[triangleIndices] .* Koi

            copyto!(I1,Symmetric(I1))
            pattern_stack[j] .= unfoldPattern(I1,pattern_stack[j])

            originimgs[centerArea,centerArea,j] .= pattern_stack[j][:,:]
        end
        return
    end

    pattern_stack = Array{Array{Complex{Float64},2},1}(undef, length(x3objspace))
    originimgs = complex(zeros(x1_imgspacelength,x1_imgspacelength,length(x3objspace)))
    #findsizes(IMGSIZE_REF, x3objmax, subNnum, centerArea_stack)
    allocateStackMem(pattern_stack, x3objspace, centerPT)
    integratePSF(originimgs, pattern_stack, x3objspace, centerPT, fobj, lambda, alpha, M, dx, k0, k, a0)

    #This is what happens when Shu Jia super resolution is applied
    if a0 > 0.0
        steps = 10
        stepz = a0/10
        Ha0 = makeHmatrix(originimgs[:,:,1],subpixelpitch,stepz,k0,lambda)
        originimgs = itrfresnel_GPU!(originimgs, Ha0, steps)
    end

    return originimgs
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




""" DEFINE OBJECT SPACE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First calculation is made at origin across the extent of the desired
 Z-axis space (i.e. zmin to zmax in intervals of zspacing)
Define point space on Z axis:

Z-axis across desired space"""
x3objspace = zmin:zspacing:zmax
"Maximum absolute value in Z-axis"
x3objmax = maximum(abs.(x3objspace))

"""Provide a surplus of values to check in one dimension + direction to
search for spread of light. In real distance this is 20*pitch of lenslet"""
x1testspace = (0:1:(subNnum*20)) * subpixelpitch

"""Calculate PSF in one dimension (find values of X at given Z distance).
Provides the halfwidth of region considered in follow-up simulation and
reconstruction."""
psfLine = calcPSFFT(x1testspace, M, k, alpha, x3objmax, fobj)

"Set a threshold where the value of the psfLine reaches near zero"
outArea = psfLine.<0.01
if sum(outArea) == 0
    error("Estimated PSF size exceeds the limit")
end
IMGSIZE_REF = cld(findfirst(outArea)[1],subNnum)
println("Maximum half-size of PSF ~= " * string(IMGSIZE_REF) * " * [microlens pitch]")
if a0 > 0
    """ Gives the number of supersampled pixels across the image. Note padding of
    1 extra microlens after ceiling value above."""
    IMG_HALFWIDTH = max( subNnum*(IMGSIZE_REF + 1), 12*subNnum) ## 12 is arbitrary
else
    IMG_HALFWIDTH = max( subNnum*(IMGSIZE_REF + 1), 2*subNnum)
end
"Create X-Y image space based on halfwidth."
x1_imgspace =(-IMG_HALFWIDTH:1:IMG_HALFWIDTH) * subpixelpitch
x2_imgspace =(-IMG_HALFWIDTH:1:IMG_HALFWIDTH) * subpixelpitch
x1_imgspacelength = length(x1_imgspace)
x2_imgspacelength = length(x2_imgspace)


#####################################################
#####################################################




"Create X-Y MLA space based on subpixels(this is just the size of a lenslet...)"
x1MLspace = (-(subNnum-1)/2 : 1 : (subNnum-1)/2) * subpixelpitch
x2MLspace = (-(subNnum-1)/2 : 1 : (subNnum-1)/2) * subpixelpitch

"Define the Microlens Array"
MLARRAY = calcML(x1_imgspace,x1_imgspacelength,x1MLspace)

""" PROJECTION FROM SINGLE POINT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define the index location of image center"""
centerPT = cld(x1_imgspacelength,2)

"Halfwidth of the maximum image (not padded)"
halfWidth = IMG_HALFWIDTH

println("Computing PSFs (1/3)")

"Execute PSF computing function"
psfWAVE_STACK = calcPSF(x3objspace,k,M,fobj,lambda,alpha,IMGSIZE_REF,subNnum,
centerPT,x1_imgspacelength,x2_imgspacelength, x3objmax, subpixelpitch, k0, a0)
end

println("Prepping for PSF Computation")
@time begin
""" Compute Light Field PSFs (light field) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Define object space imaged by each lenslet in X & Y planes. Previously this
was modeled as a line at the origin (0,0) on Z axis. Now expanded to 3D
"""
x1objspace = (pixelPitch/M).*(-fld(Nnum,2):1:fld(Nnum,2))
x2objspace = (pixelPitch/M).*(-fld(Nnum,2):1:fld(Nnum,2))

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
afgc(1000)
Ht = calcht(Himgs, Nnum, x3objspace,Xidx, Yidx, Zidx)


@save "fullSimulation.jld2" Himgs Himgs32bit M Nnum SHIFTX SHIFTY Xidx Yidx Zidx pixelPitch x3objspace
#occupied = Himgs32bit[:,1,:] .!= 0


#skimageio.imsave("Himgs32bit.tiff", Himgs32bit)
#A = skimageio.imread("foo.tiff"
