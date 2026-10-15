function [pool,data,plan] = samplePool(data,count,allowed,c,plan)
% Random 1-second speech crops. The plan alone is enough to rebuild on resume.
restore=nargin>=5 && ~isempty(plan);
if restore
    count=numel(plan.FileIndex);
else
    plan=struct('FileIndex',allowed(randi(numel(allowed),1,count)), ...
        'Start',zeros(1,count),'Length',zeros(1,count));
end
pool=struct('Audio',zeros(1,count,c.ChunkSamples,'single'),'Lengths',zeros(1,count,'uint16'));
lowEnergy=0;
for j=1:count
    k=plan.FileIndex(j);
    if isnan(data.SampleRate(k))
        a=audioinfo(data.Files(k)); data.SampleRate(k)=a.SampleRate; data.TotalSamples(k)=a.TotalSamples;
    end
    rate=data.SampleRate(k); samples=data.TotalSamples(k);
    assert(rate==c.SourceSampleRate,'S8:SampleRate','Training WAV must be 16 kHz: %s (%d Hz).',data.Files(k),rate);
    assert(samples>0,'S8:Data','Empty WAV: %s',data.Files(k));
    window=round(c.ChunkSamples*rate/c.SampleRate);
    for attempt=1:3
        if restore
            first=plan.Start(j); last=first+plan.Length(j)-1;
        else
            first=randi(max(1,samples-window+1)); last=min(samples,first+window-1);
        end
        x=audioread(data.Files(k),[first last]); x=mean(single(x),2);
        assert(all(isfinite(x)),'S8:Data','Nonfinite audio in %s',data.Files(k));
        if restore || sqrt(mean(x.^2))>=0.003 || attempt==3, break; end
    end
    plan.Start(j)=first; plan.Length(j)=last-first+1;
    if rate~=c.SampleRate, x=resample(x,c.SampleRate,rate); end
    lowEnergy=lowEnergy+(sqrt(mean(x.^2))<0.003);
    n=min(numel(x),c.ChunkSamples);
    assert(n>0,'S8:Data','Empty resampled crop: %s',data.Files(k));
    pool.Audio(1,j,1:n)=reshape(x(1:n)/max(max(abs(x)),single(0.01)),1,1,[]);
    pool.Lengths(j)=uint16(n);
    if mod(j,512)==0, fprintf('Pool: %d/%d clips\n',j,count); end
end
fprintf('Pool ready: %d random clips, %.1f MiB, %.1f%% low-energy.\n', ...
    count,numel(pool.Audio)*4/2^20,100*lowEnergy/count);
end
