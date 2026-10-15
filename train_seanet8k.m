function model = train_seanet8k(options)
% CPU/GPU training: random 16 kHz crops resampled to 8 kHz before training.
% train_seanet8k; train_seanet8k(Epochs=30,StepsPerEpoch=200,BatchSize=32)
arguments
    options.DataFolder (1,1) string = fullfile(fileparts(mfilename('fullpath')),'data')
    options.OutputFolder (1,1) string = fullfile(fileparts(mfilename('fullpath')),'models')
    options.ModelFile (1,1) string = fullfile(fileparts(mfilename('fullpath')),'model_8k.mat')
    options.InitModel (1,1) string = fullfile(fileparts(mfilename('fullpath')),'model_8k.mat')
    options.Resume (1,1) logical = true
    options.ExecutionEnvironment (1,1) string {mustBeMember(options.ExecutionEnvironment,["auto" "cpu" "gpu"])} = "auto"
    options.Epochs (1,1) double {mustBeInteger,mustBePositive} = 60
    options.StepsPerEpoch (1,1) double {mustBeInteger,mustBePositive} = 200
    options.PoolSize (1,1) double {mustBeInteger,mustBePositive} = 512
    options.ValidationClips (1,1) double {mustBeInteger,mustBePositive} = 32
    options.RefreshEvery (1,1) double {mustBeInteger,mustBePositive} = 2
    options.BatchSize (1,1) double {mustBeInteger,mustBeNonnegative} = 0
    options.LearnRate (1,1) double {mustBePositive} = 3e-4
    options.GradientClip (1,1) double {mustBeNonnegative} = 5
    options.WarmupSteps (1,1) double {mustBeInteger,mustBeNonnegative} = 20
    options.SaveEvery (1,1) double {mustBeInteger,mustBePositive} = 200
    options.MaxUpdates (1,1) double {mustBePositive} = inf
    options.Plots (1,1) logical = true
    options.Seed (1,1) double = 42
end
requested=s8.config();
if ~isfolder(options.OutputFolder), mkdir(options.OutputFolder); end
lock=s8.trainingLock(options.OutputFolder); %#ok<NASGU>
diary(fullfile(options.OutputFolder,'fast_training.log')); logGuard=onCleanup(@() diary('off'));
useGPU=options.ExecutionEnvironment~="cpu" && canUseGPU;
assert(options.ExecutionEnvironment~="gpu" || useGPU,'S8:GPU','Requested GPU is unavailable.');
if useGPU
    g=gpuDevice; deviceName=g.Name; toDevice=@gpuArray;
else
    deviceName='CPU'; toDevice=@s8.cpu;
end
rng(options.Seed); if useGPU, gpurng(options.Seed); end
checkpointFile=fullfile(options.OutputFolder,'checkpoint.mat');
stopFile=fullfile(options.OutputFolder,'STOP_TRAINING');
if isfile(stopFile), delete(stopFile); end
if options.Resume && isfile(checkpointFile)
    saved=load(checkpointFile,'model','training');

end
data=s8.catalog(options.DataFolder);
assert(numel(data.Files)>=2,'S8:Data','Use at least two WAV files for separate train/validation files.');
if options.Resume && isfile(checkpointFile)
    model=saved.model; training=saved.training;
    checkArchitecture(model.Config,requested);
    assert(isequal(data.RelativeFiles,training.Data.RelativeFiles) && isequal(data.Bytes,training.Data.Bytes),'S8:Resume','WAV dataset changed. Use a new output folder.');
    assert(options.StepsPerEpoch==training.Options.StepsPerEpoch && ...
        options.WarmupSteps==training.Options.WarmupSteps,'S8:Resume','Keep StepsPerEpoch and WarmupSteps when resuming.');
        data.SampleRate=training.Data.SampleRate; data.TotalSamples=training.Data.TotalSamples;
    rng(training.RNG);
    if useGPU && ~isempty(training.GPURNG), gpurng(training.GPURNG); end
    fprintf('Resume fast step %d, global Adam step %d.\n',training.FastStep,training.Iteration);
else
    if strlength(options.InitModel)>0 && isfile(options.InitModel)
        model=s8.loadModel(options.InitModel);
        checkArchitecture(model.Config,requested);
        fprintf('Warm start: %s (%d existing updates).\n',options.InitModel,model.TrainedSteps);
    else
        model=s8.build(requested);
    end
    order=randperm(numel(data.Files)); nval=min(128,max(1,floor(0.05*numel(order))));
    training=struct('FastVersion',1,'Epoch',1,'NextBatch',1,'FastStep',0,'Iteration',model.TrainedSteps,'OptimizerStep',0, ...
        'AverageE',[],'AverageSqE',[],'AverageD',[],'AverageSqD',[], ...
        'TrainFiles',order(nval+1:end),'ValidationFiles',order(1:nval),'Data',data, ...
        'PoolPlan',[],'PoolEpoch',0,'ValidationPlan',[],'BestVal',inf,'BestStep',-1, ...
        'History',zeros(0,9),'Completed',false);
end
training.Options=options; training.Completed=false;
summary=codec_info(model,Plot=options.Plots);
fprintf('Device: %s | %d WAV candidates | sampled training, not an exhaustive pass.\n',deviceName,numel(data.Files));
fprintf('Budget: %d epochs x %d updates; pool %d clips; refresh every %d epochs.\n', ...
    options.Epochs,options.StepsPerEpoch,options.PoolSize,options.RefreshEvery);
if ~isempty(summary.Figure), exportgraphics(summary.Figure,fullfile(options.OutputFolder,'cnn.png')); end
[validation,data,training.ValidationPlan]=s8.samplePool(data,options.ValidationClips,training.ValidationFiles,model.Config,training.ValidationPlan);
if isempty(training.PoolPlan)
    [pool,data,training.PoolPlan]=s8.samplePool(data,options.PoolSize,training.TrainFiles,model.Config);
    training.PoolEpoch=training.Epoch;
else
    [pool,data]=s8.samplePool(data,options.PoolSize,training.TrainFiles,model.Config,training.PoolPlan);
end
cache=toDevice(pool.Audio); lengths=toDevice(single(pool.Lengths)); hasPadding=any(pool.Lengths<model.Config.ChunkSamples);
if useGPU, model=s8.device(model,'gpu'); end
if options.BatchSize==0 && useGPU
    [batch,timing]=s8.tuneBatch(model,cache,[4 8 16 32]);
    writetable(timing,fullfile(options.OutputFolder,'batch_benchmark.csv'));
else
    batch=max(2,options.BatchSize);
end
fprintf('Selected batch %d; audio pool %.1f MiB.\n',batch,numel(cache)*4/2^20);
training.BatchSize=batch;
for name={'AverageE','AverageSqE','AverageD','AverageSqD'}
    if ~isempty(training.(name{1})), training.(name{1})=dlupdate(toDevice,training.(name{1})); end
end
monitor=[];
if options.Plots
    monitor=trainingProgressMonitor(Metrics=["TrainLoss","ValidationLoss"], ...
        Info=["Epoch","Batch","ClipsPerSecond","ETA","GPU"],XLabel='Training update');
    groupSubPlot(monitor,'Loss',["TrainLoss","ValidationLoss"]); monitor.Status='Training';
end
if model.VQInitialized && isinf(training.BestVal)
    training.BestVal=s8.validatePool(model,validation,batch); training.BestStep=training.Iteration;
    s8.saveRuntime(options.ModelFile,model);
    fprintf('Initial validation loss: %.4f\n',training.BestVal);
end
t=tic; recentTime=tic; recentSamples=0; speed=0;
while training.Epoch<=options.Epochs && training.FastStep<options.MaxUpdates
    if isfile(stopFile) || (~isempty(monitor) && monitor.Stop), break; end
    if training.NextBatch==1 && training.Epoch-training.PoolEpoch>=options.RefreshEvery
        [pool,data,training.PoolPlan]=s8.samplePool(data,options.PoolSize,training.TrainFiles,model.Config);
        cache=toDevice(pool.Audio); lengths=toDevice(single(pool.Lengths));
        hasPadding=any(pool.Lengths<model.Config.ChunkSamples); training.PoolEpoch=training.Epoch;
    end
    indices=randi(size(cache,2),1,batch);
    gain=0.3+0.6*rand(1,batch,1,'like',cache);
    x=dlarray(cache(:,indices,:).*gain,'CBT');
    if hasPadding
        mask=dlarray(single(reshape(toDevice(1:model.Config.ChunkSamples),1,1,[])<=reshape(lengths(indices),1,[],1)),'CBT');
    else
        mask=dlarray(ones(1,batch,model.Config.ChunkSamples,'like',cache),'CBT');
    end
    warmup=~model.VQInitialized && training.FastStep<options.WarmupSteps;
    [loss,ge,gd,model,parts]=dlfeval(@s8.gradients,model,x,warmup,mask);
    numbers=double(s8.cpu(extractdata([stripdims(loss) parts])));
    assert(all(isfinite(numbers)),'S8:Loss','Nonfinite loss; last checkpoint is preserved.');
    if options.GradientClip>0
        ge=dlupdate(@(value) clip(value,options.GradientClip),ge);
        gd=dlupdate(@(value) clip(value,options.GradientClip),gd);
    end
    rate=options.LearnRate*max(0.1,0.5*(1+cos(pi*(training.Epoch-1)/options.Epochs)));
    iteration=training.Iteration+1;
    training.OptimizerStep=training.OptimizerStep+1;
    [model.Encoder,training.AverageE,training.AverageSqE]=adamupdate(model.Encoder,ge,training.AverageE,training.AverageSqE,training.OptimizerStep,rate);
    [model.Decoder,training.AverageD,training.AverageSqD]=adamupdate(model.Decoder,gd,training.AverageD,training.AverageSqD,training.OptimizerStep,rate);
    training.Iteration=iteration; training.FastStep=training.FastStep+1; model.TrainedSteps=iteration; model.Status='training';
    recentSamples=recentSamples+batch;
    training.History(end+1,:)=[training.FastStep training.Epoch numbers(1) nan rate 0 toc(t) iteration numbers(2)];
    if mod(training.FastStep,10)==0 || training.FastStep==1
        speed=recentSamples/max(toc(recentTime),eps); recentSamples=0; recentTime=tic;
        remaining=(options.Epochs-training.Epoch)*options.StepsPerEpoch+options.StepsPerEpoch-training.NextBatch;
        eta=remaining*batch/max(speed,eps)/60;
        fprintf('Epoch %d/%d | update %d/%d | loss %.4f | %.0f clips/s | ETA %.1f min | batch %d\n', ...
            training.Epoch,options.Epochs,training.NextBatch,options.StepsPerEpoch,numbers(1),speed,eta,batch);
        training.History(end,6)=speed;
        if ~isempty(monitor)
            recordMetrics(monitor,training.FastStep,TrainLoss=numbers(1));
            updateInfo(monitor,Epoch=sprintf('%d/%d',training.Epoch,options.Epochs),Batch=batch, ...
                ClipsPerSecond=round(speed),ETA=sprintf('%.1f min',eta),GPU=deviceName);
            monitor.Progress=100*((training.Epoch-1)*options.StepsPerEpoch+training.NextBatch)/(options.Epochs*options.StepsPerEpoch);
        end
        drawnow limitrate;
    end
    training.NextBatch=training.NextBatch+1;
    endEpoch=training.NextBatch>options.StepsPerEpoch; improved=false;
    if endEpoch
        if model.VQInitialized
            value=s8.validatePool(model,validation,batch); training.History(end,4)=value;
            improved=value<training.BestVal;
            if improved, training.BestVal=value; training.BestStep=training.Iteration; end
            fprintf('Validation %.4f | best %.4f\n',value,training.BestVal);
            if ~isempty(monitor), recordMetrics(monitor,training.FastStep,ValidationLoss=value); end
        end
        training.Epoch=training.Epoch+1; training.NextBatch=1;
    end
    if mod(training.FastStep,options.SaveEvery)==0 || endEpoch
        snapshot=saveState();
        if improved, s8.saveRuntime(options.ModelFile,snapshot); end
    end
end
training.Completed=training.Epoch>options.Epochs;
if training.Completed, model.Status='completed'; else, model.Status='paused'; end
if model.VQInitialized
    value=s8.validatePool(model,validation,batch);
    if value<training.BestVal, training.BestVal=value; training.BestStep=training.Iteration; end
end
snapshot=saveState();
if training.Iteration==training.BestStep
    s8.saveRuntime(options.ModelFile,snapshot);
elseif training.BestStep>=0 && isfile(options.ModelFile)
    best=load(options.ModelFile,'model'); best.model.Status=model.Status;
    s8.saveRuntime(options.ModelFile,best.model);
end
if ~isempty(monitor), monitor.Status=model.Status; end
fprintf('%s | fast updates %d | best validation %.4f | model %s\n', ...
    upper(model.Status),training.FastStep,training.BestVal,options.ModelFile);
if ~model.VQInitialized
    fprintf('No quantized model yet: increase the update budget beyond WarmupSteps.\n');
end
model=s8.device(model,'cpu');

    function snapshot=saveState()
        state=training; state.Data=data; state.RNG=rng; state.GPURNG=[];
        if useGPU, state.GPURNG=gpurng; end
        for key={'AverageE','AverageSqE','AverageD','AverageSqD'}
            if ~isempty(state.(key{1})), state.(key{1})=dlupdate(@s8.cpu,state.(key{1})); end
        end
        snapshot=s8.saveModel(checkpointFile,model,state); model.ModelID=snapshot.ModelID;
        history=array2table(training.History,VariableNames={'FastStep','Epoch','TrainLoss','ValidationLoss','LearnRate','ClipsPerSecond','SessionSeconds','GlobalStep','WaveLoss'});
        writetable(history,fullfile(options.OutputFolder,'fast_history.csv'));
        status=struct('Status',model.Status,'Epoch',training.Epoch,'NextBatch',training.NextBatch, ...
            'FastStep',training.FastStep,'GlobalStep',training.Iteration,'BatchSize',batch, ...
            'PoolSize',size(cache,2),'StepsPerEpoch',options.StepsPerEpoch,'ClipsPerSecond',speed, ...
            'BestValidationLoss',training.BestVal,'Sampling','random 16kHz crops resampled to 8kHz; device cache');
        fid=fopen(fullfile(options.OutputFolder,'fast_status.json'),'w');
        assert(fid>=0,'S8:IO','Cannot write fast_status.json.'); closer=onCleanup(@() fclose(fid));
        fprintf(fid,'%s',jsonencode(status,PrettyPrint=true));
    end
end
function checkArchitecture(actual,requested)
for key={'SampleRate','ChunkSamples','Strides','Channels','LatentDim','NumCodebooks','CodebookSize','ResidualDilations'}
    assert(isequal(actual.(key{1}),requested.(key{1})),'S8:Architecture', ...
        'Model architecture differs at %s. Use a matching InitModel/checkpoint or a new OutputFolder.',key{1});
end
end
function g=clip(g,limit)
if ~isempty(g), g=g*min(1,limit/(sqrt(sum(g.^2,'all'))+1e-8)); end
end
