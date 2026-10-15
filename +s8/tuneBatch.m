function [best,result] = tuneBatch(model,pool,candidates)
% Time private GPU copies. Preserve CPU/GPU RNG and live training weights.
cpuState=rng; gpuState=gpurng;
guard=onCleanup(@() restoreRNG(cpuState,gpuState));
g=gpuDevice; rows=zeros(0,3); c=model.Config;
for batch=candidates
    try
        trial=model;
        x=dlarray(pool(:,mod(0:batch-1,size(pool,2))+1,:)*0.75,'CBT');
        mask=dlarray(gpuArray.ones(1,batch,c.ChunkSamples,'single'),'CBT');
        elapsed=0;
        for j=1:6
            wait(g); clock=tic;
            [loss,~,~,trial]=dlfeval(@s8.gradients,trial,x,false,mask);
            wait(g);
            assert(isfinite(double(s8.cpu(extractdata(loss)))),'S8:Tune','Nonfinite benchmark loss.');
            if j>2, elapsed=elapsed+toc(clock); end
        end
        rows(end+1,:)=[batch elapsed/4 batch*4/elapsed]; %#ok<AGROW>
        fprintf('GPU benchmark: batch %d | %.1f clips/s\n',batch,rows(end,3));
        clear trial x mask loss
    catch cause
        clear trial x mask loss
        if contains(lower(cause.message),'memory') || contains(lower(cause.identifier),'oom')
            fprintf('GPU benchmark: batch %d exceeds memory; skipped.\n',batch);
        else
            rethrow(cause);
        end
    end
end
assert(~isempty(rows),'S8:Tune','No batch size fits GPU memory.');
[~,ix]=max(rows(:,3)); best=rows(ix,1);
result=array2table(rows,VariableNames={'BatchSize','SecondsPerStep','ClipsPerSecond'});
end
function restoreRNG(cpuState,gpuState)
rng(cpuState); gpurng(gpuState);
end
