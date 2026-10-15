function model = device(model,target)
if strcmp(target,'gpu'), f = @gpuArray; else, f = @s8.cpu; end
model.Encoder = dlupdate(f,model.Encoder);
model.Decoder = dlupdate(f,model.Decoder);
for name = {'Codebooks','EMACount','EMASum'}
    model.(name{1}) = f(model.(name{1}));
end
end
