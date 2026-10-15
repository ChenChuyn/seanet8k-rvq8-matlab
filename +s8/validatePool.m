function value = validatePool(model,pool,batchSize)
% Fixed validation clips; never update weights or EMA codebooks.
c=model.Config; total=0; count=size(pool.Audio,2);
for first=1:batchSize:count
    ix=first:min(first+batchSize-1,count);
    x=dlarray(pool.Audio(:,ix,:)*0.75,'CBT');
    mask=dlarray(single(reshape(1:c.ChunkSamples,1,1,[])<=reshape(pool.Lengths(ix),1,[],1)),'CBT');
    if isa(model.Codebooks,'gpuArray'), x=gpuArray(x); mask=gpuArray(mask); end
    z=predict(model.Encoder,x); q=s8.quantize(extractdata(z),model,false);
    y=predict(model.Decoder,dlarray(q,'CBT'));
    latentMask=mask(:,:,1:c.Hop:end);
    commitment=sum((z-dlarray(q,'CBT')).^2.*latentMask,'all')/max(sum(latentMask,'all')*c.LatentDim,1);
    loss=s8.reconstructionLoss(y,x,mask,commitment,c.CommitmentWeight,c);
    total=total+double(s8.cpu(extractdata(loss)))*numel(ix);
end
value=total/count;
end
