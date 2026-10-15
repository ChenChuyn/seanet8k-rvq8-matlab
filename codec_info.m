function info = codec_info(modelFile,options)
% Print CNN dimensions/parameter counts and draw the codec architecture.
arguments
    modelFile = fullfile(fileparts(mfilename('fullpath')),'model_8k.mat')
    options.Plot (1,1) logical = true
end
if isstruct(modelFile), model=modelFile; else, model=s8.loadModel(string(modelFile)); end
c=model.Config;
encoder=sum(cellfun(@numel,model.Encoder.Learnables.Value));
decoder=sum(cellfun(@numel,model.Decoder.Learnables.Value));
vq=numel(model.Codebooks);
fprintf('\nCNN SPEECH CODEC | %d Hz mono | %d bit/s | %g frames/s\n',c.SampleRate,c.Bitrate,c.SampleRate/c.Hop);
fprintf('Parameters: encoder %s | decoder %s | RVQ %s | %.2f MiB FP32\n', ...
    char(string(encoder)),char(string(decoder)),char(string(vq)),4*(encoder+decoder+vq)/2^20);
stage=["Input";"Conv7";"Down"+string(c.Strides(:))+" + residual";"2x LSTM + skip";"Latent / RVQ"];
channels=[1 c.Channels c.Channels(end) c.LatentDim].';
samples=[c.SampleRate c.SampleRate c.SampleRate./cumprod(c.Strides) c.SampleRate/c.Hop c.SampleRate/c.Hop].';
dimensions=table(stage,channels,samples,VariableNames={'EncoderStage','Channels','TimeSamples'});
disp(dimensions);
fprintf('RVQ: %d codebooks x %d entries x %d dimensions; %d bits/frame.\n\n', ...
    c.NumCodebooks,c.CodebookSize,c.LatentDim,c.NumCodebooks*c.BitsPerIndex);
info=struct('EncoderParameters',encoder,'DecoderParameters',decoder,'CodebookParameters',vq, ...
    'Bitrate',c.Bitrate,'SampleRate',c.SampleRate,'Dimensions',dimensions,'Figure',[]);
if ~options.Plot, return; end
f=figure(Name='CNN Speech Codec',NumberTitle='off',Color='white',Position=[80 150 1100 480]);
layout=tiledlayout(f,2,2,TileSpacing='compact',Padding='compact');
ax=nexttile(layout,[1 2]);
labels={'PCM\n8 kHz',sprintf('Residual CNN\nstride %d',c.Hop),sprintf('2x LSTM\n%d',c.Channels(end)), ...
    sprintf('Latent\n%d x %g Hz',c.LatentDim,c.SampleRate/c.Hop), ...
    sprintf('RVQ\n%d x %d',c.NumCodebooks,c.CodebookSize),sprintf('2x LSTM\n%d',c.Channels(end)), ...
    sprintf('Residual CNN\nupsample %d',c.Hop),'PCM\n8 kHz'};
labels=cellfun(@(label) strrep(label,'\n',newline),labels,UniformOutput=false);
axis(ax,[0 8 0 1]); axis(ax,'off'); hold(ax,'on');
for k=1:8
    if k<=4
        color=[.14 .36 .77]; fill=[.93 .96 1];
    elseif k==5
        color=[.85 .44 .10]; fill=[1 .96 .90];
    else
        color=[.14 .55 .38]; fill=[.92 .98 .95];
    end
    rectangle(ax,Position=[k-1+.06 .30 .88 .42],Curvature=.08,EdgeColor=color,FaceColor=fill,LineWidth=1.2);
    text(ax,k-.5,.51,labels{k},HorizontalAlignment='center',VerticalAlignment='middle',FontSize=9,Interpreter='none');
    if k<8, quiver(ax,k-.05,.51,.10,0,0,Color=[.45 .5 .6],MaxHeadSize=2,LineWidth=1); end
end
title(ax,'Encoder → discrete bits → Decoder');
ax=nexttile(layout); bar(ax,[encoder decoder vq]/1e6,FaceColor=[.14 .36 .77]);
xticklabels(ax,{'Encoder','Decoder','RVQ'}); ylabel(ax,'Million parameters'); grid(ax,'on');
ax=nexttile(layout); axis(ax,'off');
text(ax,0,0.85,sprintf('Sample rate: %d Hz\nBitrate: %.1f kbit/s\nHop: %d samples\nLatent: %d channels x %g frames/s\nCNN filters: %s\nDownsample strides: %s', ...
    c.SampleRate,c.Bitrate/1000,c.Hop,c.LatentDim,c.SampleRate/c.Hop,mat2str(c.Channels),mat2str(c.Strides)), ...
    FontSize=11,VerticalAlignment='top');
info.Figure=f;
end
