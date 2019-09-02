module phasespace

% Realignment H to phase_space PSF
IMGsize=size(H,1)-mod((size(H,1)-Nnum),2*Nnum);
psf =zeros(IMGsize,IMGsize,Nnum,Nnum,size(H,5));
for z=1:size(H,5)

    LFtmp=Hlayer #without second shift
    # use A = reshape(permutedims(a, [3,2,1]), (Nnum,Nnum,lenslets,:)) to do in one line
    multiWDF=zeros(Nnum,Nnum,size(LFtmp,1)/size(H,3),size(LFtmp,2)/size(H,4),Nnum,Nnum);
    for i=1:Nnum
        for j=1:Nnum
            for a=1:lenslets
                for b=1:lenslets
                    multiWDF(i,j,a,b,:,:)=LFtmp((a-1)*Nnum+i, (b-1)*Nnum+j , :);
                end
            end
        end
    end

    WDF=zeros(  size(LFtmp,1),size(LFtmp,2),Nnum,Nnum  );
    for a=1:size(LFtmp,1)/size(H,3)
        for c=1:Nnum
            x=Nnum*a+1-c;
            for b=1:size(LFtmp,2)/size(H,4)
                for d=1:Nnum
                    y=Nnum*b+1-d;
                    WDF(x,y,:,:)=multiWDF(:,:,a,b,c,d);
                end
            end
        end
    end
    psf(:,:,:,:,z)=WDF;
end
psf_t=zeros(size(psf)); # psf_t is psf/phase space Himgs with each image rotated 180 degrees
for ii=1:Nnum
    for jj=1:Nnum
        for cc=1:size(psf,5)
            psf_t(:,:,ii,jj,cc)=rot90(squeeze(psf(:,:,ii,jj,cc)),2 );
        end
    end
end
