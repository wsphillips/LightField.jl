module projection

using FFTW
using PaddedViews

import LightField.params.ParameterSet, LightField.params.Space
import LightField.psf.fresnelH
import LightField.phasespace.lfphase!
export propagate


function propagate(originpsf::Array{Complex{Float64},3},
                     mlarray::Array{Complex{Float64},2},
                     mla::Space, img::Space, obj::Space, par::ParameterSet)

    "Image translation using PaddedViews"
    function shiftimg(img::Union{Array{Complex{Float64},2},Array{Float64,2}},
                                                         Δx::Int64, Δy::Int6)
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
                        img::Union{Array{Complex{Float64},2},Array{Float64,2}})
        Threads.@threads for i in 1:size(dest,3)
            @inbounds dest[:,:,i] .= shiftimg(img, SHIFTX[i], SHIFTY[i])
        end
    end

    function shiftimg!(dest::Union{Array{Complex{Float64},3},Array{Float64,3}},
                        img::Union{Array{Complex{Float64},3},Array{Float64,3}})
        Threads.@threads for i in 1:size(dest,3)
            @views @inbounds dest[:,:,i] .= shiftimg(img[:,:,i], SHIFTX[i], SHIFTY[i])
        end
    end

    """Calculate translation coordinates for each point in object space with respect
    to the origin"""
    function calcshifts()
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
    function sample()
        half1 = img.center:-par.sim.osr:1
        half2 = (img.center + par.sim.osr):par.sim.osr:img.xlen
        samples = vcat(half1,half2)
        sort!(samples)
        while mod(length(samples),par.sim.vpix) > 0
            global samples = samples[2:end-1]
        end
        return samples
    end

    """Makes lowpass 2D sinc kernel for filtering prior to 3x downsampling.
    This might be improved by using separable kernel"""
    function sinc1d()
        x = range(-2, stop=2, length=13) #NOTE THIS ASSUMES OSR OF 3!!!
        scale = sum(sinc.(x))
        return sinc.(x) ./ scale
    end

    function lfconvprop!()

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

        "Performs Fresnel propagation via 2D Fourier space convolution"
        function fresnelconv!(images::Array{Complex{Float64},3},
                                   H::Array{Complex{Float64},2})
            plan*images
            parimgmul!(images,H)
            plan\images
            return
        end

        function downsample!(Hsub::SubArray)
            N = par.sim.vpix
            hview = reshape(view(Hprefilt,:), (size(Hprefilt,1), size(Hprefilt,2), N, N))

            Threads.@threads for i in 1:N, j in 1:N
                a = @views conv(sinckern,sinckern, hview[:,:,i,j])[6:end-6,6:end-6]
                Hsub[:,:,i,j] = view(a,samples,samples)
            end

            return
        end

        sinckern = sinc1d()
        (SHIFTX,SHIFTY,Zidx) = calcshifts()

        for layer in 1:obj.zlen
            layerview = @view Himgs[:,:,:,:,layer]

            shiftimg!(originpsf[:,:,layer])
            parimgmul!(Himgtemp,mlarray)
            fresnelconv!(Himgtemp, H)
            parpsfmag!(Hprefilt, Himgtemp)
            downsample!(layerview)
            lfphase!(layerview)

        end

        return
    end

    "Preallocation routine for propagation image stacks"
    function initprop()
        N = par.sim.vpix
        lenslets = fld(length(samples),N)
        imgsperlayer = N^2

        H = fresnelH(originpsf[:,:,1], par, par.opt.d)
        Himgs = zeros(length(samples), length(samples), N, N, obj.zlen)
        Htemp = zeros(Complex{Float64}, size(originpsf,1), size(originpsf,1), imgsperlayer)
        Hfilt = zeros(length(samples), length(samples), imgsperlayer)
        multiWDF = zeros(N, N, lenslets, lenslets, N, N)

        FFTW.set_num_threads(fld(Threads.nthreads(),2))
        plan = plan_fft!(Htemp,[1,2], flags=FFTW.MEASURE)

        return (imgsperlayer, H, Htemp, Himgs, plan)
    end

    (imgsperlayer,H,Htemp,Himgs,plan) = initprop()
    lfconvprop!()

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

# example scratch
#= this allows logical reassignment of a view. Using the vectorized broadcast
# would instead return an allocated array.
# This modifies the parent and preserves the view + still fast to do.
for i in eachindex(viewa)
    if viewa[i] < 0.5
        viewa[i] = 0
    end
end
=#
