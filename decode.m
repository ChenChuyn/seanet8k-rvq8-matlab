function [audio,fs] = decode(model,bits)
% Decode transport bits to mono 8 kHz audio using a MAT filename/model struct.
[audio,fs]=decode_audio(model,bits);
end
