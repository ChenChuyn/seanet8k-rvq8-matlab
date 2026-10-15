function model = build(c)
% Reimplemented architecture inspired by encodec/modules/seanet.py.
g = layerGraph([sequenceInputLayer(1,Normalization='none',Name='input')
    convolution1dLayer(7,c.Channels(1),Padding='same',Name='stem')]);
tail = 'stem';
for s = 1:numel(c.Strides)
    dilations=1;
    if isfield(c,'ResidualDilations') && s>=1, dilations=c.ResidualDilations; end
    for j=1:numel(dilations)
        prefix=sprintf('er%d',s);
        if isfield(c,'ResidualDilations'), prefix=sprintf('%s_%d',prefix,j); end
        [g,tail] = residual(g,tail,c.Channels(s),prefix,dilations(j));
    end
    name = sprintf('down%d',s);
    g = addLayers(g,[eluLayer(Name=[name '_elu'])
        convolution1dLayer(2*c.Strides(s),c.Channels(s+1), ...
        Stride=c.Strides(s),Padding='same',Name=name)]);
    g = connectLayers(g,tail,[name '_elu']);
    tail = name;
end
[g,tail] = memory(g,tail,c.Channels(end));
g = addLayers(g,[eluLayer(Name='latent_elu')
    convolution1dLayer(7,c.LatentDim,Padding='same',Name='latent')]);
g = connectLayers(g,tail,'latent_elu');
model.Encoder = dlnetwork(g);
g = layerGraph([sequenceInputLayer(c.LatentDim,Normalization='none',Name='input')
    convolution1dLayer(7,c.Channels(end),Padding='same',Name='stem')]);
[g,tail] = memory(g,'stem',c.Channels(end));
for s = numel(c.Strides):-1:1
    name = sprintf('up%d',s);
    g = addLayers(g,[eluLayer(Name=[name '_elu'])
        transposedConv1dLayer(2*c.Strides(s),c.Channels(s), ...
        Stride=c.Strides(s),Cropping='same',Name=name)]);
    g = connectLayers(g,tail,[name '_elu']);
    tail=name;
    dilations=1;
    if isfield(c,'ResidualDilations') && s>=1, dilations=c.ResidualDilations; end
    for j=1:numel(dilations)
        prefix=sprintf('dr%d',s);
        if isfield(c,'ResidualDilations'), prefix=sprintf('%s_%d',prefix,j); end
        [g,tail] = residual(g,tail,c.Channels(s),prefix,dilations(j));
    end
end
g = addLayers(g,[eluLayer(Name='out_elu')
    convolution1dLayer(7,1,Padding='same',Name='wave')
    tanhLayer(Name='output')]);
g = connectLayers(g,tail,'out_elu');
model.Decoder = dlnetwork(g);
model.Config = c;
model.Codebooks = zeros(c.LatentDim,c.CodebookSize,c.NumCodebooks,'single');
model.EMACount = zeros(c.CodebookSize,c.NumCodebooks,'single');
model.EMASum = model.Codebooks;
model.VQInitialized = false;
model.ModelID = uint8(randi([0 255],1,16));
model.TrainedSteps = 0;
model.Status = 'untrained';
end

function [g,tail] = residual(g,input,channels,prefix,dilation)
g = addLayers(g,[eluLayer(Name=[prefix '_a'])
    convolution1dLayer(3,max(8,floor(channels/2)),Padding='same',DilationFactor=dilation,Name=[prefix '_c1'])
    eluLayer(Name=[prefix '_b'])
    convolution1dLayer(1,channels,Name=[prefix '_c2'])]);
g = addLayers(g,additionLayer(2,Name=[prefix '_sum']));
g = connectLayers(g,input,[prefix '_a']);
g = connectLayers(g,input,[prefix '_sum/in1']);
g = connectLayers(g,[prefix '_c2'],[prefix '_sum/in2']);
tail = [prefix '_sum'];
end

function [g,tail] = memory(g,input,channels)
g=addLayers(g,[lstmLayer(channels,OutputMode='sequence',Name='memory1')
    lstmLayer(channels,OutputMode='sequence',Name='memory2')]);
g=addLayers(g,additionLayer(2,Name='memory_skip'));
g=connectLayers(g,input,'memory1');
g=connectLayers(g,input,'memory_skip/in1');
g=connectLayers(g,'memory2','memory_skip/in2');
tail='memory_skip';
end