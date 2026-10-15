function stats = demo_codec(wavFile,options)
% Round trip a WAV, save reconstruction and transport bits for inspection.
arguments
    wavFile (1,1) string
    options.ModelFile (1,1) string = fullfile(fileparts(mfilename('fullpath')),'model_8k.mat')
    options.OutputFolder (1,1) string = fullfile(fileparts(mfilename('fullpath')),'demo_output')
    options.Play (1,1) logical = false
end
[x,fs]=audioread(wavFile);
[bits,stats]=encode_audio(options.ModelFile,x,fs);
[y,rate]=decode_audio(options.ModelFile,bits);
reference=mean(single(x),2);
if fs~=rate, reference=resample(reference,rate,fs); end
stats.SNRdB=10*log10((sum(reference.^2)+eps)/(sum((reference-y).^2)+eps));
if ~isfolder(options.OutputFolder), mkdir(options.OutputFolder); end
audiowrite(fullfile(options.OutputFolder,'reference_8k.wav'),reference,rate);
audiowrite(fullfile(options.OutputFolder,'decoded_8k.wav'),y,rate);
save(fullfile(options.OutputFolder,'encoded_bits.mat'),'bits','stats');
disp(stats);
if options.Play, sound(y,rate); end
end
