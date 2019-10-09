module projection

using FFTW, PaddedViews
using ..params, ..psf
using Base.Threads
export propagate, HMatrix

struct Delta:
  x::Vector{Int}
  y::Vector{Int}
  function Delta(lf::LightField)
    obj = lf.obj
    sim = lf.par.sim
    offset = obj.center
    xidx = repeat(1:obj.xlen, inner = obj.ylen)
    yidx = repeat(obj.ylen:1, outer = obj.xlen)
    deltax = Int.((yidx .- offset) .* sim.osr)
    deltay = Int.((yidx .- offset) .* sim.osr)
    new(deltax, deltay)
  end
end

struct HMatrix

    kernel::Array{ComplexF64,2}
    flayer::Array{ComplexF64,3}
    rlayer::Array{Float64,3}
    dsrlayer::Array{Float64,3}
    multiwdf::Array{Float64,6}
    perlayer::Int
    lenslets::Int
    N::Int
    plan::FFTWplan # TODO: extend type definition

    function HMatrix(lf::LightField, originpsf::Array{ComplexF64,3}, samples::Int)
        const p = lf.par
        const obj = lf.obj
        const N = p.sim.vpix

        lenslets = fld(samples,N)
        perlayer = N^2
        psfsize = size(originpsf,1)

        kernel = fresnelH(originpsf[:,:,1], p, p.opt.d)
       
        flayer = Threads.@spawn Array{ComplexF64,3}(undef,
                                                 (psfsize, psfsize, perlayer))
        rlayer = Threads.@spawn Array{Float64,3}(undef,
                                                 (psfsize, psfsize, perlayer))
        dsrlayer = Threads.@spawn Array{Float64,3}(undef,
                                              (samples, samples, perlayer))
        multiwdf = Threads.@spawn Array{Float64,6}(undef,
                                                  (N, N, lenslets, lenslets, N, N))
       
        FFTW.set_num_threads(Threads.nthreads()>>1))
        plan = plan_fft!(fetch(flayer),[1,2], flags=FFTW.MEASURE)

        new(kernel, flayer, fetch(rlayer), fetch(dsrlayer), fetch(multiwdf),
            perlayer, lenslets, N, plan)
    end
end

"Manually generated sampling--to ensure the center point remains fixed"
function samples(lf::LightField)
    const img = lf.imgspace
    const sim = lf.params.sim

    half1 = img.center:-sim.osr:1
    half2 = (img.center + sim.osr):sim.osr:img.xlen
    s::Vector{Int} = vcat(half1, half2)
    sort!(s)
    while mod(length(s), sim.vpix) > 0
        global s = s[2:end-1]
    end
    return s
end

"Makes 1D sinc kernel for separable LP img filter prior to 3x downsampling"
function sinc1d()
    # NOTE THIS ASSUMES OSR OF 3
    x = range(-2, stop=2, length=13)
    scale = sum(sinc.(x))
    return sinc.(x) ./ scale
end

"Image translation using PaddedViews"
function shift(img::Union{Array{ComplexF64,2},Array{Float64,2}},
                                                     Δx::Int, Δy::Int6)
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

function shiftpsf!(H::HMatrix, img::Array{ComplexF64,2}, delta::Delta)
    Threads.@threads for i in 1:size(dest,3)
        @inbounds dest[:,:,i] .= shift(img, delta.x[i], delta.y[i])
    end
end

function stackmul!(imgs::Union{Array{ComplexF64,3},Array{Float64,3}},
                   kernel::Union{Array{ComplexF64,2},Array{Float64,2}})
    Threads.@threads for i in 1:size(imgs,3)
        @views @inbounds imgs[:,:,i] .= imgs[:,:,i] .* kernel
    end
end

function psfmag!(dest::Array{Float64,3}, psfimgs::Array{ComplexF64,3})
    Threads.@threads for i in 1:size(dest,3)
        @views @inbounds dest[:,:,i] .= abs2.(psfimgs[:,:,i])
    end
end

function fresnelconv!(images::Array{ComplexF64,3},
                      H::Array{ComplexF64,2})
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

function propagate(originpsf::Array{ComplexF64,3}, lf::LightField, mlarray::Array{ComplexF64,2})
    sinckern = sinc1d()
    delta = Delta(lf)
    samples = samples(lf)
    Hlayer = HMatrix(lf, originpsf, samples)
    # make H matrix
    for layer in 1:obj.zlen

        shiftpsf!(Hlayer, originpsf[:,:,layer])
        stackmul!(Hlayer, mlarray)
        fresnelconv!(Hlayer)
        parpsfmag!(Hlayer)
        downsample!(Hlayer)
        lfphase!(Hlayer)
    end
    return
end

function lfconvprop!()
    # REPLACE WITH Type constructor
    #(imgsperlayer,H,Htemp,Himgs,plan) = initprop()
    lfconvprop!()

    return Himgs
end

#=

function shiftstack!(dest::Union{Array{ComplexF64,3},Array{Float64,3}},
                    img::Union{Array{ComplexF64,3},Array{Float64,3}})
    Threads.@threads for i in 1:size(dest,3)
        @views @inbounds dest[:,:,i] .= shift(img[:,:,i], delta.x[i], delta.y[i])
    end
end

function postprocess(Himgs::Array{ComplexF64,3}, objspace::Space, Zidx::Array{Int,1})

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
