function [bits,stats] = encode_audio(modelFile,audio,fs)
% Encode waveform to a logical column vector for transmission/modulation.
arguments
    modelFile
    audio {mustBeNumeric,mustBeReal,mustBeFinite,mustBeNonempty}
    fs (1,1) double {mustBePositive,mustBeFinite,mustBeInteger} = 8000
end
model=s8.loadModel(modelFile);
[bytes,stats]=s8.encode(model,audio,fs);
bits=reshape(logical(bsxfun(@bitget,bytes(:),8:-1:1).'),[],1);
end
