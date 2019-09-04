module projection

using FFTW
using PaddedViews
import LightField.psf.fresnelH
import LightField.params.ParameterSet, LightField.params.Space
export propagate

function parimgmul!(imgs::Union{Array{Complex{Float64},3},Array{Float64,3}}, kernel::Union{Array{Complex{Float64},2},Array{Float64,2}})

    Threads.@threads for i in 1:size(imgs,3)
        @views @inbounds imgs[:,:,i] .= imgs[:,:,i] .* kernel
    end
end

function parpsfmag!(dest::Array{Float64,3}, psfimgs::Array{Complex{Float64},3})
    Threads.@threads for i in 1:size(dest,3)
        @views @inbounds dest[:,:,i] .= abs2.(psfimgs[:,:,i])
    end
end
"Image translation using PaddedViews"
function shiftimg(img::Union{Array{Complex{Float64},2},Array{Float64,2}},
    Δx::Int64, Δy::Int64)
    @views begin
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
end

function shiftimg!(dest::Union{Array{Complex{Float64},3},Array{Float64,3}},
                   img::Union{Array{Complex{Float64},2},Array{Float64,2}},
                            Δx::Array{Int64,1}, Δy::Array{Int64,1})
    Threads.@threads for i in 1:size(dest,3)
        @inbounds dest[:,:,i] .= shiftimg(img, Δx[i], Δy[i])
    end
end

function shiftimg!(dest::Union{Array{Complex{Float64},3},Array{Float64,3}},
                   img::Union{Array{Complex{Float64},3},Array{Float64,3}},
                            Δx::Array{Int64,1}, Δy::Array{Int64,1})
    Threads.@threads for i in 1:size(dest,3)
        @views @inbounds dest[:,:,i] .= shiftimg(img[:,:,i], Δx[i], Δy[i])
    end
end

"""Calculate translation coordinates for each point in object space with respect
to the origin"""
function calcshifts(obj::Space, par::ParameterSet)
    "These are scaling terms to convert array subscripts to obj space"
    XREF = obj.center
    YREF = XREF
    Xidx = repeat(1:obj.xlen, inner=obj.ylen)
    Yidx = repeat(obj.ylen:1, outer=obj.xlen)
    Zidx = repeat(1:obj.zlen, inner=(obj.xlen*obj.ylen))

    "For shifting the images relative to object space"
    SHIFTX = Int64.((Xidx.-XREF).*par.sim.osr)
    SHIFTY = Int64.((Yidx.-YREF).*par.sim.osr)
    return (SHIFTX, SHIFTY, Zidx)
end

"Manual sampling points--to ensure the center point remains consistent"
function sample(img::Space, par::ParameterSet)
    half1 = img.center:-par.sim.osr:1
    half2 = (img.center + par.sim.osr):par.sim.osr:img.xlen
    samples = vcat(half1,half2)
    return sort!(samples)
end
"Makes lowpass 2D sinc kernel for filtering prior to 3x downsampling"
function sinc2d()
    sinc2d(x,y) = (sinc(x)*sinc(y))
    x1 = range(-2, stop=2, length=13) #NOTE THIS ASSUMES OSR OF 3!!!
    x2 = x1'
    return sinc2d.(x1,x2).*0.330
end

"Preallocation routine for propagation image stacks"
function initprop(originpsf::Array{Complex{Float64},3},
                                      mla::Space,obj::Space,par::ParameterSet)
    imgsperlayer = par.sim.vpix^2
    H = fresnelH(originpsf[:,:,1], par, par.opt.d)
    Himgs = zeros(cld(size(originpsf,1),par.sim.osr), cld(size(originpsf,1),par.sim.osr), imgsperlayer*obj.zlen)
    Himgtemp = zeros(Complex{Float64}, size(originpsf,1), size(originpsf,1), imgsperlayer)

    FFTW.set_num_threads(fld(Threads.nthreads(),2))
    plan = plan_fft!(Himgtemp,[1,2], flags=FFTW.MEASURE)

    return (imgsperlayer, H, Himgtemp, Himgs, plan)
end

"Performs Fresnel propagation via 2D Fourier space convolution"
function fresnelconv!(plan, images::Array{Complex{Float64},3},
                            H::Array{Complex{Float64},2})
    plan*images
    parimgmul!(images,H)
    plan\images
    return
end

"Returns linear indices corresponding to all XY images within each Z-layer"
function chunks(layer::Int64, imgsperlayer::Int64)
    cstart = 1 + (layer-1)*imgsperlayer
    cend   = layer*imgsperlayer
    return (cstart, cend)
end

function lfconvprop!(originpsf::Array{Complex{Float64},3},
                     mlarray::Array{Complex{Float64},2},
                     SHIFTX::Array{Int64,1}, SHIFTY::Array{Int64,1},
                     img::Space, obj::Space, par::ParameterSet, imgsperlayer::Int64,
                     H::Array{Complex{Float64},2},
                     Himgtemp::Array{Complex{Float64},3},
                     Himgs::Array{Float64,3}, plan::FFTW.cFFTWPlan{Complex{Float64},-1,true,3})

        samples = sample(img, par)
    for layer in 1:obj.zlen
        #TODO: layer "chunks" deprecated in favor of tradition 5D matrix
        shiftimg!(Himgtemp,originpsf[:,:,layer],SHIFTX,SHIFTY)
        parimgmul!(Himgtemp,mlarray)
        fresnelconv!(plan, Himgtemp, H)
        #sinc filter..via FFT before power conversion???
        #abs2 of images via parpsfmag!()
        #downsample
        #phase space conversion via index reassignment
    end
    return
end

function propagate(originpsf::Array{Complex{Float64},3},
                     mlarray::Array{Complex{Float64},2},
                     mla::Space, img::Space, obj::Space, par::ParameterSet)

    (SHIFTX,SHIFTY,Zidx) = calcshifts(obj,par)
    (imgsperlayer,H,Himgtemp,Himgs,plan) = initprop(originpsf,mla,obj,par)
    lfconvprop!(originpsf, mlarray, SHIFTX, SHIFTY, img, obj, par, imgsperlayer, H,
                                                    Himgtemp, Himgs, plan)

    # phasespace()
    # postprocess()

    return Himgs
end

#=
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
=#

end # end projection module
