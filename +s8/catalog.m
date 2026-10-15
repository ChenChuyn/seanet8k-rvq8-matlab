function data = catalog(root)
% Relative names let the same dataset/checkpoint move to another laptop.
assert(isfolder(root),'S8:Data','DataFolder does not exist: %s',root);
[ok,attrib]=fileattrib(root); assert(ok,'S8:Data','Cannot resolve DataFolder.');
root=string(attrib.Name);
files=dir(fullfile(root,'**','*'));
files=files(~[files.isdir]);
names=string({files.name}); files=files(endsWith(names,'.wav',IgnoreCase=true));
assert(~isempty(files),'S8:Data','No WAV under %s. Pass DataFolder or put WAV in data/.',root);
paths=string(fullfile({files.folder},{files.name})).';
relative=extractAfter(paths,strlength(root)+1);
relative=replace(relative,'\','/');
[relative,order]=sort(relative); files=files(order); paths=paths(order);
data=struct('Files',paths,'RelativeFiles',relative,'Bytes',[files.bytes].', ...
    'SampleRate',nan(numel(files),1),'TotalSamples',nan(numel(files),1));
end
