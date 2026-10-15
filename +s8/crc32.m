function crc = crc32(bytes)
% IEEE CRC-32, no Java or Communications Toolbox dependency.
persistent table
if isempty(table)
    table=zeros(256,1,'uint32');
    for k=0:255
        r=uint32(k);
        for j=1:8
            if bitand(r,1), r=bitxor(bitshift(r,-1),uint32(hex2dec('EDB88320')));
            else, r=bitshift(r,-1); end
        end
        table(k+1)=r;
    end
end
crc=uint32(hex2dec('FFFFFFFF'));
for b=reshape(uint8(bytes),1,[])
    crc=bitxor(bitshift(crc,-8),table(double(bitand(bitxor(crc,uint32(b)),255))+1));
end
crc=bitcmp(crc);
end
