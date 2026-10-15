function [loss,ge,gd,model,parts] = gradients(model,x,warmup,mask)
if nargin<4, mask=x*0+1; end
z = forward(model.Encoder,x);
if warmup
    zq = z;
    commitment = sum(z.^2,'all')*0;
else
    latentMask=mask(:,:,1:model.Config.Hop:end);
    [q,~,model] = s8.quantize(extractdata(z),model,true,logical(extractdata(latentMask)));
    % Straight-through estimator: identity encoder gradient, discrete forward.
    zq = z + dlarray(q-extractdata(z),'CBT');
    commitment = sum((z-dlarray(q,'CBT')).^2.*latentMask,'all') ...
        /max(sum(latentMask,'all')*model.Config.LatentDim,1);
end
y = forward(model.Decoder,zq);
[loss,parts]=s8.reconstructionLoss(y,x,mask,commitment,model.Config.CommitmentWeight,model.Config);
[ge,gd] = dlgradient(loss,model.Encoder.Learnables,model.Decoder.Learnables);
end
