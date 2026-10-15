function out = packet(action,varargin)
% Wire format, big endian integers, MSB-first bits:
% S8R1[4], modelID[16], nSamples[u32], gain[f32], payloadBytes[u32],
% Model-defined B-bit RVQ indices (frame-major), zero pad, CRC32[u32].
switch action
    case 'pack'
        [codes,model,n,gain]=deal(varargin{:});
        c=model.Config;
        values=reshape(uint16(codes),1,[]);
        bits=reshape(bsxfun(@bitget,values(:),c.BitsPerIndex:-1:1).',1,[]);
        bits=[bits zeros(1,mod(-numel(bits),8),'uint16')];
        payload=uint8(double(reshape(bits,8,[]).')*(2.^(7:-1:0)).');
        gainbits=typecast(single(gain),'uint32');
        header=[uint8('S8R1') reshape(model.ModelID,1,[]) put32(uint32(n)) ...
            put32(gainbits) put32(uint32(numel(payload)))];
        bytes=[header reshape(payload,1,[])];
        out=[bytes put32(s8.crc32(bytes))];
    case 'unpack'
        bytes=reshape(uint8(varargin{1}),1,[]); model=varargin{2}; c=model.Config;
        assert(numel(bytes)>=36 && isequal(bytes(1:4),uint8('S8R1')), ...
            'S8:Packet','Invalid or truncated packet.');
        assert(isequal(bytes(5:20),reshape(model.ModelID,1,[])), ...
            'S8:WrongModel','Bitstream and MAT model do not match.');
        n=double(get32(bytes(21:24)));
        gain=typecast(get32(bytes(25:28)),'single');
        len=double(get32(bytes(29:32)));
        frames=ceil(n/c.Hop);
        assert(n>0 && n<=c.SampleRate*3600 && isfinite(gain) && gain>0 && gain<=2, ...
            'S8:Packet','Invalid sample count or gain (maximum 1 hour).');
        assert(len==ceil(frames*c.NumCodebooks*c.BitsPerIndex/8) && numel(bytes)==36+len, ...
            'S8:Packet','Packet length is incorrect.');
        assert(s8.crc32(bytes(1:end-4))==get32(bytes(end-3:end)), ...
            'S8:CRC','Packet checksum failed: bits were changed or lost.');
        bits=reshape(bsxfun(@bitget,bytes(33:end-4).',8:-1:1).',1,[]);
        count=frames*c.NumCodebooks*c.BitsPerIndex;
        assert(all(bits(count+1:end)==0),'S8:Packet','Nonzero padding bits.');
        bits=reshape(bits(1:count),c.BitsPerIndex,[]);
        values=uint16((2.^(c.BitsPerIndex-1:-1:0))*double(bits));
        out=struct('Codes',reshape(values,c.NumCodebooks,frames),'Samples',n,'Gain',gain);
    case 'bits'
        bytes=reshape(uint8(varargin{1}),[],1);
        out=char(reshape(bsxfun(@bitget,bytes,8:-1:1).',1,[])+'0');
    case 'bytes'
        txt=regexprep(char(varargin{1}),'\s','');
        assert(~isempty(txt) && all(txt=='0' | txt=='1') && mod(numel(txt),8)==0, ...
            'S8:Bits','Enter only 0/1 bits and whitespace; length must be a multiple of 8.');
        out=uint8((2.^(7:-1:0))*double(reshape(txt-'0',8,[])));
    otherwise
        error('S8:Packet','Unknown packet operation.');
end
end

function b=put32(v)
b=uint8(bitand(bitshift(v,[-24 -16 -8 0]),255));
end
function v=get32(b)
v=bitor(bitor(bitshift(uint32(b(1)),24),bitshift(uint32(b(2)),16)), ...
    bitor(bitshift(uint32(b(3)),8),uint32(b(4))));
end
