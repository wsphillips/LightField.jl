module phasespace

#something like this

function lfphase!(Himgtemp::AbstractArray, par::ParameterSet)

    multiWDF = zeros(Nnumy, Nnumx, lensletsy, lensletsx, Nnum, Nnum)
    for i in 1:par.sim.vpix, j in 1:par.sim.vpix
        bylenslets = @view(Himgtemp[i:par.sim.vpix:end, j:par.sim.vpix:end, :])
        for a in 1:lenslets, b in 1: lenslets
            multiWDF[i, j, a, b, :, :] .= reshape(@view(bylenslets[a,b,:]), (par.sim.vpix, par.sim.vpix))
        end
    end

    for a in 1:lenslets, i in 1:Nnum
            x = Nnum * a + 1 - i;
            for b in 1:lenslets, j in 1:Nnum
                    y = Nnum * b + 1 - j;
                    @views Himgtemp[x, y, :, :] .= multiWDF[:, :, a, b, i, j];
            end
    end

    for i in 1:par.sim.vpix, j in 1:par.sim.vpix
        Himgtemp[:,:,i,j] .= rot180(Himgtemp[:,:,i,j])
    end

    Himgs[:,:,:,:,z] .= Himgtemp

end
