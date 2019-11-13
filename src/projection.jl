module projection

using FFTW, PaddedViews
using ..params, ..psf 
using Base.Threads
using DSP
#import FFTW.cFFTWplan
export propagate, HMatrix

struct Delta
  x::Vector{Int}
  y::Vector{Int}
  function Delta(lf::LightFieldSimulation)
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
    dsrlayer::Array{Float32,3}
    multiwdf::Array{Float64,6}
    perlayer::Int
    lenslets::Int
    N::Int
    plan #::FFTW.cFFTWplan{Complex{Float64},-1,true,3}
    function HMatrix(lf::LightFieldSimulation, originpsf::Array{ComplexF64,3}, samples::Int)
        p = lf.par
        obj = lf.obj
        N = p.sim.vpix
        lenslets = fld(samples,N)
        perlayer = N^2
        psfsize = size(originpsf,1)

        kernel = fresnelH(originpsf[:,:,1], p, p.opt.d)
       
        flayer = Array{ComplexF64,3}(undef,(psfsize, psfsize, perlayer))
        rlayer = Array{Float64,3}(undef,(psfsize, psfsize, perlayer))
        dsrlayer = Array{Float32,3}(undef,(samples, samples, perlayer))
        multiwdf = Array{Float32,6}(undef,(N, N, lenslets, lenslets, N, N))
       
        FFTW.set_num_threads(Threads.nthreads()>>1)
        plan = plan_fft!(flayer,[1,2], flags=FFTW.MEASURE)

        new(kernel, flayer, rlayer, dsrlayer, multiwdf, perlayer, lenslets, N, plan)
    end
end

"Manually generated sampling--to ensure the center point remains fixed"
function samples(lf::LightFieldSimulation)
    img = lf.img
    sim = lf.par.sim

    half1 = img.center:-sim.osr:1
    half2 = (img.center + sim.osr):sim.osr:img.xlen
    s::Vector{Int} = vcat(half1, half2)
    sort!(s)
    # TODO: fix scope shit here
    while mod(length(s), sim.vpix) > 0
        s = s[2:end-1]
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
function shift(img::Union{Array{ComplexF64,2},Array{Float64,2}}, Δx::Int, Δy::Int)
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
    Threads.@threads for i in 1:size(H.flayer,3)
        @inbounds H.flayer[:,:,i] .= shift(img, delta.x[i], delta.y[i])
    end
end

function stackmul!(H::HMatrix, tf::Array{ComplexF64,2})
    Threads.@threads for i in 1:size(H.flayer,3)
        @views @inbounds H.flayer[:,:,i] .= H.flayer[:,:,i] .* tf
    end
end

function psfmag!(H::HMatrix)
    Threads.@threads for i in 1:size(H.flayer,3)
        @views @inbounds H.rlayer[:,:,i] .= abs2.(H.flayer[:,:,i])
    end
end

function fresnelconv!(H::HMatrix)
    H.plan*H.flayer
    stackmul!(H, H.kernel)
    H.plan\H.flayer
    return
end

function downsample!(H::HMatrix, kernel1D::Vector{Float64}, samples::Vector{Int})
    Threads.@threads for i in 1:size(H.rlayer,3)
        a = @view(conv(kernel1D,kernel1D, H.rlayer[:,:,i])[6:end-6,6:end-6])
        H.dsrlayer[:,:,i] .= view(a,samples,samples)
    end
    return
end

function lfphase!(H::HMatrix, lf::LightFieldSimulation)
  N = lf.par.sim.vpix
  zlen = lf.obj.zlen
  samplelen = size(H.dsrlayer,1)
  lenslets = H.lenslets 
  
  for i in 1:N, j in 1:N
    bylenslets = @view(H.dsrlayer[i:N:end, j:N:end, :])
    for a in 1:lenslets, b in 1:lenslets
      H.multiwdf[i, j, a, b, :, :] .= reshape(@view(bylenslets[a,b,:]), (N,N))
    end
  end
  layer4d = reshape(view(H.dsrlayer,:), samplelen, samplelen, N, N)
  for a in 1:lenslets, i in 1:N
    x = N * a + 1 - i;
    for b in 1:lenslets, j in 1:N
      y = N * b + 1 - j;
      @views layer4d[x, y, :, :] .= H.multiwdf[:, :, a, b, i, j];
    end
  end

  for i in 1:N, j in 1:N
    layer4d[:,:,i,j] .= rot180(layer4d[:,:,i,j])
  end

  return 
end

function propagate(lf::LightFieldSimulation)
    # call to generate originpsf
    originpsf = originPSFproj(lf)
    obj = lf.obj
    sinckern = sinc1d()
    delta = Delta(lf)
    samplepts = samples(lf)
    Hlayer = HMatrix(lf, originpsf, length(samplepts))
    Himgs = Array{Float32,5}(undef,length(samplepts), length(samplepts), Hlayer.N, Hlayer.N, obj.zlen)
    # make H matrix
    for layer in 1:obj.zlen

        shiftpsf!(Hlayer, originpsf[:,:,layer], delta)
        stackmul!(Hlayer, lf.mla.array)
        fresnelconv!(Hlayer)
        psfmag!(Hlayer)
        downsample!(Hlayer, sinckern, samplepts)
        #lfphase!(Hlayer, lf)
        Himgs[:,:,:,:,layer] .= reshape(view(Hlayer.dsrlayer,:),length(samplepts),length(samplepts), lf.par.sim.vpix, lf.par.sim.vpix)
    end
    return Himgs
end

end #module

#=

function postprocess(Himgs::Array{ComplexF64,3}, objspace::Space, Zidx::Array{Int,1})
    #TODO: Needs to be revised for new pipeline

    "Normalize the whole data set"
    Himgs = Himgs./maximum(Himgs)

    """Threshold all the images (grouped by Z-plane) by the tolerance
    value. Flooring all values below the cutoff to zero."""
    tol = 0.005


    for layer in 1:lf.obj.zlen
        H4Dslice = Himgs[:,:,Zidx .== layer]
        H4Dslice[H4Dslice .< (tol*maximum(H4Dslice))] .= 0
        Himgs[:,:,Zidx .== layer] = H4Dslice
    end
    return Himgs
end
=#

