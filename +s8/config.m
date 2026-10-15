function c = config()
% Fixed portable 8 kHz SEANet variant, not pretrained EnCodec weights.
c.FormatVersion=1;
c.Profile='seanet8k_rvq8';
c.SampleRate=8000;
c.SourceSampleRate=16000;
c.Strides=[2 4 5 8];
c.Hop=prod(c.Strides);
c.Channels=[16 32 64 128 256];
c.LatentDim=128;
c.NumCodebooks=8;
c.CodebookSize=256;
c.BitsPerIndex=8;
c.Bitrate=c.SampleRate/c.Hop*c.NumCodebooks*c.BitsPerIndex;
c.ChunkSamples=8000;
c.ResidualDilations=[1 3 9];
c.EMADecay=0.99;
c.CommitmentWeight=0.25;
c.ZeroCodeword=false;
c.LossFFTLengths=[128 256 512 1024];
c.DifferenceWeight=0.5;
c.Architecture='SEANet variant: residual Conv1D, 2-layer skip LSTM, 128D RVQ';
c.Mode='Offline independent 1-second blocks; symmetric zero padding; CPU inference';
c.Preprocessing=struct('Channels','Mean across channels', ...
    'Resampling','Anti-aliasing resample: 16 kHz to 8 kHz before training', ...
    'Gain','One peak gain per utterance, stored in packet', ...
    'Padding','Zero pad to Hop, trim decoded output to original 8 kHz sample count');
c.Bitstream='S8R1; big endian; frame-major 8 uint8 indices; 36-byte header/CRC';
end
