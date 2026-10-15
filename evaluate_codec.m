function metrics = evaluate_codec(dataFolder,options)
% Evaluate a separate WAV directory; save paired 8 kHz WAV and objective metrics.
arguments
    dataFolder (1,1) string
    options.ModelFile (1,1) string = fullfile(fileparts(mfilename('fullpath')),'model_8k.mat')
    options.OutputFolder (1,1) string = fullfile(fileparts(mfilename('fullpath')),'evaluation')
    options.MaxFiles (1,1) double {mustBePositive,mustBeInteger} = 8
    options.Seconds (1,1) double {mustBePositive,mustBeFinite} = 5
end
data=s8.catalog(dataFolder); count=min(options.MaxFiles,numel(data.Files));
if ~isfolder(options.OutputFolder), mkdir(options.OutputFolder); end
rows=zeros(count,5);
for k=1:count
    info=audioinfo(data.Files(k));
    last=min(info.TotalSamples,max(1,round(options.Seconds*info.SampleRate)));
    [x,fs]=audioread(data.Files(k),[1 last]);
    [bits,stats]=encode_audio(options.ModelFile,x,fs);
    [y,rate]=decode_audio(options.ModelFile,bits);
    x=mean(single(x),2); if fs~=rate, x=resample(x,rate,fs); end
    x=double(x); y=double(y);
    snr=10*log10((sum(x.^2)+eps)/(sum((x-y).^2)+eps));
    centeredX=x-mean(x); centeredY=y-mean(y);
    target=centeredX*(dot(centeredX,centeredY)/(sum(centeredX.^2)+eps));
    sisdr=10*log10((sum(target.^2)+eps)/(sum((centeredY-target).^2)+eps));
    rows(k,:)=[numel(x)/rate snr sisdr mean(abs(x-y)) stats.ActualBitrate];
    audiowrite(fullfile(options.OutputFolder,sprintf('%02d_reference_8k.wav',k)),x,rate);
    audiowrite(fullfile(options.OutputFolder,sprintf('%02d_decoded_8k.wav',k)),y,rate);
end
metrics=array2table(rows,VariableNames={'Seconds','SNRdB','SISDRdB','MAE','PacketBitrate'});
metrics.File=data.RelativeFiles(1:count);
writetable(metrics,fullfile(options.OutputFolder,'metrics.csv'));
summary=struct('Files',count,'SNRdB',mean(rows(:,2)),'SISDRdB',mean(rows(:,3)), ...
    'MAE',mean(rows(:,4)),'PacketBitrate',mean(rows(:,5)),'PayloadBitrate',1600, ...
    'HumanListeningValidated',false);
fid=fopen(fullfile(options.OutputFolder,'summary.json'),'w'); assert(fid>=0,'S8:IO','Cannot write evaluation summary.');
guard=onCleanup(@() fclose(fid)); fprintf(fid,'%s',jsonencode(summary,PrettyPrint=true));
disp(summary);
end
