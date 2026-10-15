function report = test_seanet8k(options)
arguments
    options.RunGPU (1,1) logical = false
end
root=fileparts(fileparts(mfilename('fullpath'))); addpath(root);
state=rng; rngGuard=onCleanup(@() rng(state)); rng(17);
work=tempname; mkdir(work); cleanup=onCleanup(@() rmdir(work,'s'));
c=s8.config(); assert(c.Hop==320 && c.Bitrate==1600 && c.NumCodebooks==8);
m=s8.build(c); z=randn(128,2,25,'single');
[~,~,m]=s8.quantize(z,m,true);
[q,codes]=s8.quantize(z,m,false); reconstructed=zeros(128,50,'single');
assert(all(codes(:)<=255));
for k=1:8, reconstructed=reconstructed+m.Codebooks(:,double(codes(k,:))+1,k); end
assert(max(abs(q(:)-reconstructed(:)))<1e-5);
% Synthetic model is exclusively for transport tests, never exported as trained.
m.TrainedSteps=1; m.Status='synthetic test fixture';
fixture=fullfile(work,'fixture.mat'); s8.saveRuntime(fixture,m);
for n=[1 319 320 321 7999 8000 8001 16001]
    x=single(.2*sin(2*pi*230*(0:n-1)'/8000));
    [bits,stats]=encode_audio(fixture,x,8000); [y,fs]=decode_audio(fixture,bits);
    assert(islogical(bits)&&iscolumn(bits)&&numel(bits)==288+ceil(n/320)*64);
    assert(fs==8000&&numel(y)==n&&all(isfinite(y))&&stats.PayloadBitrate==1600);
end
% Public compatibility API accepts both a MAT filename and a model struct.
assert(isequal(encode(fixture,x,8000),bits));
assert(isequal(encode(s8.loadModel(fixture),x,8000),bits));
[compatAudio,compatRate]=decode(fixture,bits);
assert(compatRate==8000 && isequal(compatAudio,y));
bad=bits; bad(265)=~bad(265); mustFail(@() decode_audio(fixture,bad),'S8:CRC');
bad=bits; bad(33)=~bad(33); mustFail(@() decode_audio(fixture,bad),'S8:WrongModel');
mustFail(@() decode_audio(fixture,[0;2]),'S8:Bits');
data=fullfile(work,'data'); mkdir(data); t=(0:15999)'/16000;
for k=1:4
    audiowrite(fullfile(data,sprintf('%d.wav',k)),.2*sin(2*pi*(130+40*k)*t),16000);
end
catalog=s8.catalog(data);
[pool,~,~]=s8.samplePool(catalog,2,1:4,c);
assert(size(pool.Audio,3)==8000&&all(pool.Lengths==8000));
out=fullfile(work,'models'); trained=fullfile(work,'trained.mat');
args={'DataFolder',data,'OutputFolder',out,'ModelFile',trained,'InitModel',"", ...
    'ExecutionEnvironment',"cpu",'Epochs',1,'StepsPerEpoch',3,'PoolSize',4, ...
    'ValidationClips',2,'BatchSize',2,'WarmupSteps',1,'SaveEvery',1,'Plots',false};
train_seanet8k(args{:},MaxUpdates=2);
before=load(fullfile(out,'checkpoint.mat'),'model','training');
assert(before.training.OptimizerStep==2&&before.model.VQInitialized);
% Relocate both source WAV and checkpoint, just as on another laptop.
moved=fullfile(work,'moved'); mkdir(moved);
copyfile(data,fullfile(moved,'data')); copyfile(out,fullfile(moved,'models'));
args{2}=fullfile(moved,'data'); args{4}=fullfile(moved,'models');
train_seanet8k(args{:},MaxUpdates=3);
after=load(fullfile(moved,'models','checkpoint.mat'),'model','training');
assert(after.training.Completed&&after.training.OptimizerStep==3);
assert(isequal(before.training.PoolPlan,after.training.PoolPlan));
assert(isequal(before.training.ValidationPlan,after.training.ValidationPlan));
assert(isempty(intersect(after.training.TrainFiles,after.training.ValidationFiles)));
assert(any(before.model.Encoder.Learnables.Value{1}~=after.model.Encoder.Learnables.Value{1},'all'));
assert(any(before.model.Decoder.Learnables.Value{1}~=after.model.Decoder.Learnables.Value{1},'all'));
loaded=s8.loadModel(trained);
assert(all(isfinite(loaded.Codebooks),'all'));
[y,rate]=decode_audio(trained,encode_audio(trained,single(sin(t)),16000));
assert(rate==8000&&numel(y)==8000&&all(isfinite(y)));
portable=fullfile(work,'portable');
export_bundle(portable,ModelFile=trained,CheckpointFolder=fullfile(moved,'models'),MakeZip=true);
assert(isfile(portable+".zip")&&isfile(fullfile(portable,'model_8k.mat')));
if options.RunGPU
    m=s8.device(s8.build(c),'gpu');
    x=dlarray(gpuArray(randn(1,2,8000,'single')*.1),'CBT');
    [loss,ge,gd,m]=dlfeval(@s8.gradients,m,x,false);
    assert(isfinite(s8.cpu(extractdata(loss)))&&m.VQInitialized);
    assert(all(cellfun(@(v) all(isfinite(s8.cpu(extractdata(v))),'all'),ge.Value)));
    assert(all(cellfun(@(v) all(isfinite(s8.cpu(extractdata(v))),'all'),gd.Value)));
end
report=struct('Passed',true,'CPUTrainAndRelocatedResume',true, ...
    'PortableBundle',true,'RunGPU',options.RunGPU,'PerceptualQualityValidated',false);
disp(report);
end

function mustFail(action,id)
caught=false;
try
    action();
catch cause
    assert(strcmp(cause.identifier,id),'Expected %s, got %s.',id,cause.identifier); caught=true;
end
assert(caught,'Expected %s.',id);
end
