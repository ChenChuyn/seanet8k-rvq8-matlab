function [loss,parts] = reconstructionLoss(y,x,mask,commitment,weight,c)
if nargin<6, c=struct; end
lengths=[256 512 1024];
if isfield(c,'LossFFTLengths'), lengths=c.LossFFTLengths; end
y=real(y);
wave=sum(abs(y-x).*mask,'all')/max(sum(mask,'all'),1);
detail=wave*0;
if isfield(c,'DifferenceWeight')
    valid=mask(:,:,2:end).*mask(:,:,1:end-1);
    delta=(y(:,:,2:end)-y(:,:,1:end-1))-(x(:,:,2:end)-x(:,:,1:end-1));
    detail=sum(abs(delta).*valid,'all')/max(sum(valid,'all'),1);
    detail=detail*c.DifferenceWeight;
end
x=x.*mask; y=y.*mask; spectral=wave*0;
for n=lengths
    window=single(hann(n,'periodic'));
    a=abs(dlstft(x,Window=window,OverlapLength=3*n/4,FFTLength=n));
    b=abs(dlstft(y,Window=window,OverlapLength=3*n/4,FFTLength=n));
    spectral=spectral+mean(abs(log(a+1e-3)-log(b+1e-3)),'all')/numel(lengths) ...
        +sqrt(sum((a-b).^2,'all')+1e-6)/sqrt(sum(a.^2,'all')+1e-6)/numel(lengths);
end
loss=wave+spectral+detail+weight*commitment;
parts=[stripdims(wave) stripdims(spectral) stripdims(commitment)];
end
