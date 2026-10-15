# SEANet 8 kHz + RVQ 8 × 256 — MATLAB

Repo kèm `model_8k.mat` (CNN + 8 codebook) và `models/checkpoint.mat` để resume.
**Model hiện mới train 400 update, chất lượng thoại còn kém; cần train thêm.**
Chi tiết thực nghiệm xem [TRAINING_STATUS.md](TRAINING_STATUS.md).

Clone repo trên laptop khác:

```powershell
git clone https://github.com/ChenChuyn/seanet8k-rvq8-matlab.git
cd seanet8k-rvq8-matlab
```

Mở thư mục vừa clone trong MATLAB, chạy `setup_codec`, rồi dùng
`encode("model_8k.mat",5)` để thu microphone hoặc truyền WAV vào `encode_audio`.
WAV train không nằm trong repo; đặt trong `data/` hoặc truyền `DataFolder` tới
dataset sẵn có. Xem các lệnh train và resume bên dưới.

Gói độc lập với các thư mục codec cũ. Dùng MATLAB R2025b (phiên bản đã kiểm tra),
Deep Learning Toolbox và Signal Processing Toolbox. GPU NVIDIA và Parallel Computing
Toolbox giúp train nhanh hơn; encode/decode và train CPU đều được hỗ trợ.
Không cần Python, EnCodec, tải weights hay kết nối mạng.

## Thông số

| Thành phần | Giá trị |
|---|---|
| WAV train | 16.000 Hz, stereo được lấy trung bình thành mono |
| Đầu vào mạng / đầu ra codec | Mono 8.000 Hz, `resample` có lọc chống alias |
| Stride encoder | `[2 4 5 8]`, tổng 320 mẫu |
| Latent | 128 chiều × 25 frame/giây |
| RVQ | 8 codebook; mỗi codebook 256 vector 128 chiều |
| Index | 0–255; 1 index 8 bit/codebook/frame |
| Payload | 25 × 8 × 8 = **1.600 bit/s** |
| Packet | Thêm 36 byte header/CRC, padding cuối theo frame |

Một giây có 200 byte payload, packet 236 byte = 1.888 bit/s.
40 ms là bước frame latent; xử lý offline theo block 1 giây, không phải độ trễ streaming.
Đầu ra 8 kHz không khôi phục dải tần 4–8 kHz đã loại khi downsample.

Encoder: Conv1D kernel 7 → 4 mức residual + strided Conv1D → 2 LSTM có skip
→ Conv1D latent. Mỗi mức có residual dilation `[1 3 9]`, bottleneck /2,
ELU, channels `[16 32 64 128 256]`. Decoder đối xứng dùng transposed Conv1D
và tanh đầu ra. Đây là biến thể SEANet nhỏ cho laptop: zero padding đối xứng,
không có weight normalization; không tương thích trực tiếp weights EnCodec.
RVQ chọn nearest neighbor trên phần dư, cập nhật codebook bằng EMA, khởi tạo
từ latent, thay code chết; encoder dùng straight-through gradient.
Loss: L1 waveform + sai phân waveform + multi-resolution STFT + commitment.

## Chạy trên laptop có dữ liệu

Giải nén/copy toàn bộ thư mục, mở thư mục này trong MATLAB:

```matlab
setup_codec
% Đặt WAV 16 kHz trong data/ và các thư mục con, rồi:
train_seanet8k

% Hoặc dùng dataset ở nơi khác, không cần copy dữ liệu:
train_seanet8k(DataFolder="D:\SpeechData\raw", BatchSize=16)
% Chạy bằng CPU:
train_seanet8k(DataFolder="D:\SpeechData\raw", ...
    ExecutionEnvironment="cpu", BatchSize=2, Plots=false)
```

Ít nhất 2 file WAV; trainer báo lỗi nếu file được lấy mẫu không phải 16 kHz.
Train lấy crop 1 giây ngẫu nhiên ở sample rate gốc rồi downsample xuống 8 kHz
trước khi đưa vào mạng. Không sửa WAV gốc. Pool được refresh định kỳ, có augmentation
biên độ, mask cho file ngắn. Validation dùng file riêng, crop cố định, không cập nhật
codebook. Chia theo file, chưa chia theo người nói; đánh giá khả năng tổng quát hóa
cần tập test/nhóm người nói độc lập.

Mặc định 60 epoch × 200 update; một epoch là ngân sách update, không phải một lượt
duyệt toàn bộ WAV. Tự chọn batch trên GPU; CPU mặc định batch 2. `MaxUpdates`
là giới hạn tổng update trong checkpoint, không phải số update cộng thêm mỗi lần gọi.
Ví dụ kiểm tra nhanh với thư mục riêng:

```matlab
train_seanet8k(DataFolder="D:\SpeechData\raw", OutputFolder="quick_models", ...
    ModelFile="quick_model.mat", InitModel="", Epochs=1, StepsPerEpoch=30, ...
    WarmupSteps=5, MaxUpdates=30, PoolSize=32, ValidationClips=4, ...
    BatchSize=4, Plots=false)
```

Chỉ có model encode/decode sau khi đã qua warmup và có quantized update.
`model_8k.mat` chứa encoder, decoder, 8 codebook, config, ModelID và source snapshot
của model có validation tốt nhất. `models/checkpoint.mat` chứa trạng thái mới nhất,
Adam, EMA, RNG, split dữ liệu, crop plan để tiếp tục train. Hai snapshot có thể khác bước.
Loss/log/status nằm trong `models/fast_*`. Dừng bằng nút Stop của Training Progress
hoặc tạo file `models/STOP_TRAINING`; trainer lưu checkpoint trước khi thoát vòng lặp.
Lỗi/Ctrl+C đột ngột chỉ khôi phục được checkpoint đã lưu gần nhất.

Gọi lại cùng lệnh để resume; giữ `StepsPerEpoch` và `WarmupSteps` của checkpoint.
Khi chuyển laptop, truyền `DataFolder` mới trỏ tới cùng dataset, giữ tên/đường dẫn
tương đối và nội dung WAV. Checkpoint so sánh danh sách tương đối và kích thước file,
không phụ thuộc ký tự ổ đĩa. Không xác minh hash toàn bộ dataset.
Muốn train từ đầu dùng thư mục output mới và `InitModel=""`.

## Encode/decode sau train

Thu microphone 5 giây ở 8 kHz, nén rồi giải mã và phát lại:

```matlab
setup_codec
model = "model_8k.mat";
seconds = 5;
bits = encode(model,seconds);
[audio,fs] = decode(model,bits);
sound(audio,fs);
```

Có thể chạy trực tiếp `demo_microphone`. `model` nhận tên file MAT hoặc struct
model trả về từ trainer. `encode(model,seconds)` thu mono 8 kHz/16 bit bằng
microphone mặc định và chờ thu xong; thời lượng từ 0,1 đến 120 giây.
`encode(model,x,fs)` nhận waveform có sẵn. Dùng `setup_codec` và mở đúng thư mục
codec này để MATLAB chọn `encode.m`/`decode.m` của codec 8 kHz.

Với WAV có sẵn:

```matlab
setup_codec
[x,fs] = audioread("speech.wav");
[bits,stats] = encode_audio("model_8k.mat",x,fs);
[y,fsOut] = decode_audio("model_8k.mat",bits);
audiowrite("decoded_8k.wav",y,fsOut);
codec_info("model_8k.mat",Plot=true)
demo_codec("speech.wav")
evaluate_codec("D:\SpeechData\test",MaxFiles=8,Seconds=5)
```

`bits` là vector cột logical 0/1, MSB trước, dùng được làm đầu vào điều chế.
Giải mã kiểm tra CRC và ModelID; hai máy phải dùng **cùng snapshot model**.
Bitstream chứa số mẫu 8 kHz và gain, không chứa weights/codebook.
Decode trả vector cột single và `fsOut=8000`, trim đúng chiều dài sau resample.
Không cần dataset/GPU để encode/decode. Các API `encode_audio`/`decode_audio`
vẫn có thể dùng khi muốn tránh trùng tên `encode`/`decode` với codec khác.

## Đóng gói cho laptop khác

Sau train, chọn tên thư mục đích mới nằm ngoài thư mục gói hiện tại:

```matlab
export_bundle("D:\Transfer\SEANet8k_ready")
% Chỉ cần codec, bỏ checkpoint training:
export_bundle("D:\Transfer\SEANet8k_inference",IncludeCheckpoint=false)
```

Lệnh tạo thư mục và ZIP gồm mã nguồn, README, tests, `model_8k.mat`, `bundle.json`,
và checkpoint nếu bật. Tự kiểm tra model đã copy encode ra đúng cùng bitstream và
giải mã được bằng CPU. Dataset không nằm trong ZIP; máy kia dùng dữ liệu sẵn có
qua `DataFolder`. Muốn mang cả dữ liệu có thể tự copy WAV vào `data/` sau khi giải nén.
Không cần các thư mục `Speech_Neural_Codec` hay `encodec`.

## Kiểm thử

```matlab
addpath("tests")
test_seanet8k                  % CPU: RVQ, packet, gradient/train, resume sau chuyển data
test_seanet8k(RunGPU=true)     % thêm một bước gradient GPU
```

Các kiểm thử xác nhận pipeline, không khẳng định chất lượng nghe. Model train ngắn
chỉ là bản khởi đầu; cần train đủ và nghe/đánh giá trên speech test độc lập.
Xem `TRAINING_STATUS.md` nếu gói có sẵn kết quả chạy thực nghiệm.

Mã MATLAB được tách và điều chỉnh từ các helper trong `Speech_Neural_Codec` của
workspace này; cấu trúc mạng được đối chiếu với `encodec/modules/seanet.py` cục bộ.
