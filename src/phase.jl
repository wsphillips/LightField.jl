module phasespace

#something like this

function lfphase!(Himgtemp::AbstractArray, Himgtransformed::AbstractArray, multiWDF::AbstractArray, par::ParameterSet)

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
                    @views Himgtransformed[x, y, :, :] .= multiWDF[:, :, a, b, i, j];
            end
    end

    Threads.@threads for i in 1:N, j in 1:N
        Himgtransformed[:,:,i,j] .= rot180(Himgtransformed[:,:,i,j])
    end

    return
end
