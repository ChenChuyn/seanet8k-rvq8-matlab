function saveRuntime(path,model)
% A single portable MAT containing CPU weights, codebooks and configuration.
model=s8.device(model,'cpu');
folder=fileparts(path); if isempty(folder), folder=pwd; end
if ~isfolder(folder), mkdir(folder); end
tmp=[tempname(folder) '.mat']; cleanup=onCleanup(@() removeTemporary(tmp));
save(tmp,'model','-v7.3'); movefile(tmp,path,'f');
end
function removeTemporary(path)
if isfile(path), delete(path); end
end
