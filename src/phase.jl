module phasespace

using Threads
using ..params
export lfphase!

#TODO: Fix type declarations
function phaseindex(lf::LightField)
  N = lf.par.sim.N
  lookup::Vector{UInt64} = 1:(length(samples)*length(samples)*N*N*lf.par.obj.zlen)
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
    Himgtform[:,:,i,j] .= rot180(Himgtform[:,:,i,j])
  end
  return
end
