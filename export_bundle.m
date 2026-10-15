function folder = export_bundle(folder,options)
% Copy an immutable CPU model with all source, optionally training state.
arguments
    folder (1,1) string
    options.ModelFile (1,1) string = fullfile(fileparts(mfilename('fullpath')),'model_8k.mat')
    options.IncludeCheckpoint (1,1) logical = true
    options.CheckpointFolder (1,1) string = fullfile(fileparts(mfilename('fullpath')),'models')
    options.MakeZip (1,1) logical = true
end
root=fileparts(mfilename('fullpath'));
model=s8.loadModel(options.ModelFile);
assert(~isfolder(folder) && ~isfile(folder),'S8:Export','Choose a new destination folder: %s',folder);
assert(~options.MakeZip || ~isfile(folder+".zip"),'S8:Export','ZIP already exists: %s',folder+".zip");
mkdir(folder);
[~,a]=fileattrib(folder); folder=string(a.Name);
source=dir(fullfile(root,'*.m'));
for k=1:numel(source), copyfile(fullfile(root,source(k).name),folder); end
copyfile(fullfile(root,'+s8'),fullfile(folder,'+s8'));
copyfile(fullfile(root,'tests'),fullfile(folder,'tests'));
docs=dir(fullfile(root,'*.md'));
for k=1:numel(docs), copyfile(fullfile(root,docs(k).name),folder); end
if isfile(fullfile(root,'architecture.png')), copyfile(fullfile(root,'architecture.png'),folder); end
mkdir(fullfile(folder,'data')); mkdir(fullfile(folder,'models'));
s8.saveRuntime(fullfile(folder,'model_8k.mat'),model);
if options.IncludeCheckpoint
    file=fullfile(options.CheckpointFolder,'checkpoint.mat');
    assert(isfile(file),'S8:Export','Checkpoint not found: %s. Set IncludeCheckpoint=false for inference only.',file);
    saved=load(file,'model');
    assert(isequal(saved.model.Config,model.Config),'S8:Export','Checkpoint and model configurations differ.');
    copyfile(file,fullfile(folder,'models','checkpoint.mat'));
    for name=["fast_history.csv" "fast_status.json"]
        file=fullfile(options.CheckpointFolder,name);
        if isfile(file), copyfile(file,fullfile(folder,'models',name)); end
    end
end
manifest=struct('Format','SEANet8k_RVQ8','SampleRate',8000,'Hop',320, ...
    'Codebooks',8,'EntriesPerCodebook',256,'LatentDim',128,'PayloadBitrate',1600, ...
    'ModelID',sprintf('%02x',model.ModelID),'TrainedSteps',model.TrainedSteps, ...
    'IncludesCheckpoint',options.IncludeCheckpoint,'DatasetIncluded',false, ...
    'MATLABRelease',version('-release'));
fid=fopen(fullfile(folder,'bundle.json'),'w'); assert(fid>=0,'S8:IO','Cannot write bundle.json.');
guard=onCleanup(@() fclose(fid)); fprintf(fid,'%s',jsonencode(manifest,PrettyPrint=true)); clear guard;
% Check the copied snapshot really produces the same bitstream and waveform.
x=single(.1*sin(2*pi*190*(0:8799)'/8000));
[a,~]=encode_audio(options.ModelFile,x,8000);
[b,~]=encode_audio(fullfile(folder,'model_8k.mat'),x,8000);
assert(isequal(a,b),'S8:Export','Copied model changes encoded output.');
[y,fs]=decode_audio(fullfile(folder,'model_8k.mat'),b);
assert(fs==8000 && numel(y)==numel(x) && all(isfinite(y)),'S8:Export','Copied decoder failed.');
if options.MakeZip
    zipPath=folder+".zip";
    zip(zipPath,{'*'},folder);
end
fprintf('Portable bundle: %s\nModel step: %d; GPU/data not required for inference.\n',folder,model.TrainedSteps);
end
