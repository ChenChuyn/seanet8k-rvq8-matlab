function [audio,fs] = decode_audio(modelFile,bits)
% Decode on CPU using the identical model snapshot as the encoder.
arguments
    modelFile
    bits
end
assert((isnumeric(bits)||islogical(bits)) && isvector(bits) && ~isempty(bits) && ...
    isreal(bits) && all(isfinite(bits(:))) && all(bits(:)==0|bits(:)==1) && mod(numel(bits),8)==0, ...
    'S8:Bits','bits must be a 0/1 vector with length divisible by 8.');
bytes=uint8((2.^(7:-1:0))*double(reshape(bits,8,[])));
model=s8.loadModel(modelFile);
audio=s8.decode(model,bytes); fs=model.Config.SampleRate;
end
