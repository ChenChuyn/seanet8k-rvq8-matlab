function [bytes,stats] = encode(model,x,fs)
validateattributes(x,{'numeric'},{'nonempty','real','finite'});
if isrow(x), x=x.'; end
x=mean(single(x),2);
if fs~=model.Config.SampleRate, x=resample(x,model.Config.SampleRate,fs); end
assert(numel(x)<=model.Config.SampleRate*3600,'S8:Length','Maximum recording length is one hour.');
c=model.Config; n=numel(x);
assert(max(abs(x))<=1.01,'S8:Audio','Audio must be in the range [-1,1].');
gain=max(max(abs(x))/0.9,single(0.01));
x=x/gain;
x(end+1:ceil(n/c.Hop)*c.Hop)=0;
codes=zeros(c.NumCodebooks,numel(x)/c.Hop,'uint16');
for start=1:c.ChunkSamples:numel(x)
    stop=min(start+c.ChunkSamples-1,numel(x));
    z=predict(model.Encoder,dlarray(reshape(x(start:stop),1,1,[]),'CBT'));
    [~,ix]=s8.quantize(extractdata(z),model,false);
    first=(start-1)/c.Hop+1;
    codes(:,first:first+size(ix,2)-1)=ix;
end
bytes=s8.packet('pack',codes,model,n,gain);
stats=struct('Duration',n/c.SampleRate,'PayloadBitrate',c.Bitrate, ...
    'TotalBits',numel(bytes)*8,'ActualBitrate',numel(bytes)*8/(n/c.SampleRate), ...
    'PCM16CompressionRatio',n*16/(numel(bytes)*8));
end
