function model = saveModel(path,model,training)
% Atomic replacement; store CPU weights for machines without a GPU.
model = s8.device(model,'cpu');
% Each weight snapshot needs a distinct ID, including intermediate checkpoints.
state=rng;
model.ModelID=uint8(randi([0 255],1,16));
rng(state);
model.SavedAt = char(datetime('now',Format='yyyy-MM-dd HH:mm:ss'));
model.Runtime = struct('MATLAB',version,'Platform',computer, ...
    'Toolboxes','MATLAB, Deep Learning Toolbox, Signal Processing Toolbox');
% Keep a source snapshot in the MAT file for reproducibility.
root = fileparts(fileparts(mfilename('fullpath')));
files = [dir(fullfile(root,'+s8','*.m')); dir(fullfile(root,'*.m')); ...
    dir(fullfile(root,'*.ps1')); dir(fullfile(root,'README.md'))];
model.Source = struct('Name',{},'Text',{});
for k=1:numel(files)
    model.Source(k).Name = strrep(fullfile(files(k).folder,files(k).name),[root filesep],'');
    model.Source(k).Text = fileread(fullfile(files(k).folder,files(k).name));
end
folder = fileparts(path);
if ~isempty(folder) && ~isfolder(folder), mkdir(folder); end
if isempty(folder), folder=pwd; end
tmp = [tempname(folder) '.mat'];
cleanup = onCleanup(@() removeTemporary(tmp));
save(tmp,'model','training','-v7.3');
movefile(tmp,path,'f');
end

function removeTemporary(p)
if isfile(p), delete(p); end
end
