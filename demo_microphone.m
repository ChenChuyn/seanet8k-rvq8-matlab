% Run this script to record 5 seconds, encode, decode, and play the result.
setup_codec;
model=fullfile(fileparts(mfilename('fullpath')),'model_8k.mat');
seconds=5;
bits=encode(model,seconds);
[audio,fs]=decode(model,bits);
sound(audio,fs);
