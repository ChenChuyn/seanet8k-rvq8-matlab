function [bits,stats] = encode(model,input,fs)
% encode(model,seconds): record mono 8 kHz speech, then encode.
% encode(model,audio,fs): encode an existing waveform without recording.
% model can be a MAT filename or the trained model struct.
arguments
    model
    input {mustBeNumeric,mustBeReal,mustBeFinite,mustBeNonempty}
    fs (1,1) double {mustBePositive,mustBeFinite,mustBeInteger} = 8000
end
model=s8.loadModel(model);
if nargin==2 && isscalar(input)
    validateattributes(input,{'numeric'},{'scalar','>=',0.1,'<=',120},mfilename,'seconds');
    recorder=audiorecorder(8000,16,1);
    fprintf('Recording %.1f seconds at 8000 Hz...\n',double(input));
    recordblocking(recorder,double(input));
    audio=getaudiodata(recorder,'single');
    assert(~isempty(audio),'S8:Recording','The microphone returned no audio.');
    fs=8000;
else
    audio=input;
end
[bits,stats]=encode_audio(model,audio,fs);
end
