function value = cpu(value)
% Avoid requiring a GPU toolbox operation when the value is already on CPU.
if isa(value,'dlarray')
    if isa(extractdata(value),'gpuArray'), value=gather(value); end
elseif isa(value,'gpuArray')
    value=gather(value);
end
end
