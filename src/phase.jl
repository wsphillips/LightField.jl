module phasespace

using Threads
using ..params

export lfphase!

#TODO: Fix type declarations
function lfphase!(Himgtemp::AbstractArray, Himgtform::AbstractArray, multiWDF::AbstractArray, par::ParameterSet, img::Space)

    N = par.sim.vpix
    Threads.@threads for i in 1:N, j in 1:N
        bylenslets = @view(Himgtemp[i:N:end, j:N:end, :])
        for a in 1:lenslets, b in 1: lenslets
            multiWDF[i, j, a, b, :, :] .= reshape(@view(bylenslets[a,b,:]), (N,N))
        end
    end

    Threads.@threads for a in 1:lenslets, i in 1:N
            x = N * a + 1 - i;
            for b in 1:lenslets, j in 1:N
                    y = N * b + 1 - j;
                    @views Himgtform[x, y, :, :] .= multiWDF[:, :, a, b, i, j];
            end
    end

    Threads.@threads for i in 1:N, j in 1:N
        Himgtransform[:,:,i,j] .= rot180(Himgtransformed[:,:,i,j])
    end

    return
end
