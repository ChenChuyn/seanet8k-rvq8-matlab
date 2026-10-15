function root = setup_codec()
% Run once after opening this folder on either laptop.
root=fileparts(mfilename('fullpath')); addpath(root);
assert(~isempty(ver('nnet')),'S8:Toolbox','Install Deep Learning Toolbox.');
assert(~isempty(ver('signal')),'S8:Toolbox','Install Signal Processing Toolbox.');
fprintf('SEANet 8 kHz | hop 320 | RVQ 8 x 256 | payload 1600 bit/s\n');
fprintf('Root: %s\n',root);
end
