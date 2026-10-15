function x = decode(model,bytes)
p=s8.packet('unpack',bytes,model); c=model.Config;
frames=size(p.Codes,2); block=c.ChunkSamples/c.Hop;
x=zeros(frames*c.Hop,1,'single');
for start=1:block:frames
    stop=min(start+block-1,frames);
    q=zeros(c.LatentDim,1,stop-start+1,'single');
    for k=1:c.NumCodebooks
        q=q+reshape(model.Codebooks(:,double(p.Codes(k,start:stop))+1,k), ...
            c.LatentDim,1,[]);
    end
    y=predict(model.Decoder,dlarray(q,'CBT'));
    samples=(start-1)*c.Hop+(1:(stop-start+1)*c.Hop);
    x(samples)=reshape(s8.cpu(extractdata(y)),[],1)*p.Gain;
end
x=max(-1,min(1,x(1:p.Samples)));
end
