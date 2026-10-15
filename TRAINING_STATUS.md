# Trạng thái ngày 15/10/2026

Đã train từ đầu **400 update**, không sử dụng weights/codebook của codec cũ.
20 bước đầu warmup autoencoder, tiếp theo train qua RVQ. Đây là model thực nghiệm
ban đầu, **chưa hoàn tất ngân sách 12.000 update và chưa đạt chất lượng thoại**.

- Nguồn: 11.660 WAV ứng viên trong `Speech_Neural_Codec/data/raw`; không đọc `data/test` khi train.
- WAV 16 kHz → lấy crop 1 giây → mono → lọc/resample 8 kHz → CNN/RVQ.
- GPU: RTX 3050 Ti Laptop, batch 16, pool 512, 32 crop validation riêng.
- Validation loss: 2,2794 ở bước 200; **2,0994 ở bước 400**.
- Tốc độ các bước cuối khoảng 52 crop/giây; đây là đo ngắn, không bảo đảm thời gian tổng.
- Best model và checkpoint hiện cùng bước 400; checkpoint ở đầu epoch 3/60.

Đánh giá 8 file riêng từ `data/test`, tối đa 5 giây/file, resample cùng cách:

| Trung bình | Kết quả |
|---|---:|
| Payload | 1.600 bit/s |
| Packet, gồm header/padding | 1.687,91 bit/s |
| SNR | -0,7385 dB |
| SI-SDR | -37,5373 dB |
| MAE | 0,03651 |

Các chỉ số waveform còn kém. Loss giảm chưa chứng minh chất lượng nghe;
cần train thêm, nghe A/B, và đánh giá lại. Không xem model này là codec đã đạt
chất lượng sử dụng. CSV và WAV nghe thử nằm trong `evaluation/` của thư mục phát triển.

## Tiếp tục trên laptop hiện tại

Mở thư mục `SEANet8k_RVQ8` trong MATLAB:

```matlab
setup_codec
train_seanet8k(DataFolder="../Speech_Neural_Codec/data/raw",BatchSize=16)
```

## Tiếp tục trên laptop khác

Giải nén gói portable, mở thư mục vừa giải nén trong MATLAB:

```matlab
setup_codec
train_seanet8k(DataFolder="D:\Dataset\raw",BatchSize=16)
% Nếu chỉ có CPU:
train_seanet8k(DataFolder="D:\Dataset\raw", ...
    ExecutionEnvironment="cpu",BatchSize=2,Plots=false)
```

Để resume phải dùng cùng tập WAV với tên/đường dẫn tương đối giống dataset cũ.
Nếu dùng dataset khác, tạo checkpoint/model mới:

```matlab
train_seanet8k(DataFolder="D:\DatasetMoi",OutputFolder="models_new", ...
    ModelFile="model_new.mat",Resume=false,InitModel="model_8k.mat")
```

Lệnh này warm start CNN/codebook từ bước 400, tạo split/Adam mới cho dataset mới.
Muốn random initialization hoàn toàn, thay `InitModel=""`.

## Xác minh

Kiểm thử CPU + GPU đã qua: shape mạng, RVQ 8 tầng/index 0–255, khôi phục latent
từ index, bitstream/CRC/ModelID, chiều dài tại biên frame/block, downsample,
train với gradient cập nhật encoder/decoder, checkpoint/Adam, resume sau chuyển
dataset/checkpoint sang thư mục khác, export ZIP và CPU round trip model đã copy.
MATLAB Code Analyzer: 0 thông báo trên mã nguồn đã kiểm tra.

`model_8k.mat` lưu weights CPU để encode/decode trên máy không có GPU.
GPU chỉ được chọn khi [canUseGPU](https://www.mathworks.com/help/matlab/ref/canusegpu.html)
xác nhận thiết bị và toolbox phù hợp; có thể ép chạy CPU bằng `ExecutionEnvironment="cpu"`.
