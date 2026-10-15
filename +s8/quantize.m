function [q,codes,model] = quantize(z,model,update,valid)
% z and q are ordinary numeric C-by-B-by-T arrays, on CPU or GPU.
c = model.Config;
zeroCodeword=isfield(c,'ZeroCodeword') && c.ZeroCodeword;
shape = size(z);
r = reshape(z,c.LatentDim,[]);
if nargin<4, valid=true(1,size(r,2)); else, valid=reshape(valid,1,[]); end
q = zeros(size(r),'like',r);
gpuUpdate=update && isa(r,'gpuArray');
emitCodes=nargout>=2;
if ~emitCodes
    codes=[];
elseif gpuUpdate
    codes=gpuArray.zeros(c.NumCodebooks,size(r,2),'uint16');
else
    codes = zeros(c.NumCodebooks,size(r,2),'uint16');
end
if gpuUpdate, bins=cast((1:c.CodebookSize).','like',r); end
for k = 1:c.NumCodebooks
    e = model.Codebooks(:,:,k);
    samples=r(:,valid);
    if update && ~model.VQInitialized
        ix = randi(size(samples,2),1,c.CodebookSize);
        e = samples(:,ix) + 1e-3*randn(size(e),'like',e);
        model.EMASum(:,:,k) = e;
        model.EMACount(:,k) = 1;
    end
    % A permanent zero entry permits a stage to leave the residual unchanged.
    if zeroCodeword, e(:,1)=0; end
    distance = sum(e.^2,1).' + sum(r.^2,1) - 2*(e.'*r);
    [~,ix] = min(distance,[],1);
    chosen = e(:,ix);
    if ~emitCodes
        % Validation only needs quantized vectors, with no device-to-host codes.
    elseif gpuUpdate
        codes(k,:) = uint16(ix-1); %#ok<AGROW> Preallocated when emitCodes is true.
    else
        codes(k,:) = uint16(s8.cpu(ix)-1); %#ok<AGROW> Preallocated when emitCodes is true.
    end
    if update
        % Assignment shape: CodebookSize by the number of valid latent frames.
        active=ix(valid);
        if gpuUpdate
            % Keep assignments and EMA on the GPU; avoid three device syncs/stage.
            assignment=single(bins==active);
        else
            assignment = full(sparse(double(active),1:numel(active),1,c.CodebookSize,numel(active)));
            assignment = cast(assignment,'like',r);
        end
        count = sum(assignment,2);
        total = samples*assignment.';
        a = c.EMADecay;
        model.EMACount(:,k) = a*model.EMACount(:,k)+(1-a)*count;
        model.EMASum(:,:,k) = a*model.EMASum(:,:,k)+(1-a)*total;
        e = model.EMASum(:,:,k)./max(model.EMACount(:,k).',1e-5);
        dead = model.EMACount(:,k)<0.05;
        if gpuUpdate
            % Masked replacement avoids copying a dead-code count to the CPU.
            replacements=samples(:,randi(size(samples,2),1,c.CodebookSize,'like',r));
            mask=single(dead.');
            e=e.*(1-mask)+replacements.*mask;
            model.EMASum(:,:,k)=model.EMASum(:,:,k).*(1-mask)+replacements.*mask;
            model.EMACount(:,k)=model.EMACount(:,k).*(1-mask.')+mask.';
        else
            n=sum(dead);
            if n>0
                e(:,dead) = samples(:,randi(size(samples,2),1,n));
                model.EMACount(dead,k) = 1;
                model.EMASum(:,dead,k) = e(:,dead);
            end
        end
        if zeroCodeword
            e(:,1)=0; model.EMASum(:,1,k)=0; model.EMACount(1,k)=1;
        end
        model.Codebooks(:,:,k) = e;
    end
    q = q+chosen;
    r = r-chosen;
end
if update, model.VQInitialized = true; end
q = reshape(q,shape);
end
