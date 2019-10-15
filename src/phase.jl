module phasespace

using Base.Threads
using ..params
import ..projection.HMatrix
export lfphase!

function lfphase!(H::HMatrix, lf::LightField)
  N = lf.par.sim.N
  zlen = lf.par.obj.zlen
  samplelen = size(H.dsrlayer,1)
  lenslets = H.lenslets 
  
  Threads.@threads for i in 1:N, j in 1:N
    bylenslets = @view(H.dsrlayer[i:N:end, j:N:end, :])
    for a in 1:lenslets, b in 1:lenslets
      H.multiwdf[i, j, a, b, :, :] .= reshape(@view(bylenslets[a,b,:]), (N,N))
    end
  end

  Threads.@threads for a in 1:lenslets, i in 1:N
    x = N * a + 1 - i;
    for b in 1:lenslets, j in 1:N
      y = N * b + 1 - j;
      @views H.dsrlayer[x, y, :, :] .= H.multiwdf[:, :, a, b, i, j];
    end
  end

  Threads.@threads for i in 1:N, j in 1:N
    H.dsrlayer[:,:,i,j] .= rot180(H.dsrlayer[:,:,i,j])
  end

  return 
end
