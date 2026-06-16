# Deepfake Detection — Tiền xử lý & Trích đặc trưng (phần A + B)

`deepfake_pj_group.ipynb` lo phần **Dữ liệu (A)** và **Đặc trưng (B)** trên dataset `project_data` (20.000 ảnh real + 20.000 ảnh fake), xuất các file `.pkl` để **bàn giao cho người C** train model. Phần model (C) nằm ở notebook riêng.

## Luồng xử lý

```
ảnh → resize 256×256 (xám) → ảnh dư PCA (bỏ 32 thành phần chính)
    → FFT (64) + LBP (59) + Noise (64) → feature vector 187 chiều
    → StandardScaler → split 80/10/10 → lưu train/val/test.pkl + scaler.pkl
```

## Phân công

| Người | Nhiệm vụ | Trong file này |
|---|---|---|
| **A — Data** | Lấy `project_data`, giải nén, resize, chia 80/10/10 | ✅ có |
| **B — Features** | FFT + LBP + Noise → ghép feature vector + nhóm đặc trưng cho ablation | ✅ có |
| **C — Model** | Train SVM/RF/GBM, đánh giá, ablation | ➡️ bàn giao (file `.pkl`) |

## Những gì đã sửa ở phần Features (người B)

1. **Sửa lỗi LBP.** Bản cũ dùng `method='uniform'` (chỉ sinh ~10 mẫu) nhưng lại đặt `LBP_BINS=59`, khiến **49/59 chiều luôn bằng 0** (đặc trưng chết). Đã đổi sang `method='nri_uniform'` (đúng 59 mẫu) → dùng đủ 59 bins.
2. **`feature_groups` để ablation đúng.** Lưu vị trí cột của từng nhóm `{'fft':(0,64), 'lbp':(64,123), 'noise':(123,187)}` vào mỗi file `.pkl`. Đây là phần B *chuẩn bị dữ liệu* cho ablation; việc chạy model trên từng nhóm là của người C.
3. **Chuẩn hóa không rò rỉ dữ liệu.** `StandardScaler` chỉ `fit` trên train rồi `transform` val/test.
4. **`FEATURE_DIM` tính tự động** từ số bins (không hardcode 187) + `assert` kiểm tra chiều vector.
5. **Lấy dữ liệu bền hơn:** thử Drive đã mount trước, không thấy thì tải bằng `gdown`.

## File bàn giao B → C

`train.pkl`, `val.pkl`, `test.pkl` (mỗi file gồm `X`, `y`, `paths`, `feature_names`, `feature_groups`, `config`) và `scaler.pkl`.

Người C nạp các file này để train SVM/RF/GBM, đánh giá AUC/F1/Confusion và chạy ablation (dùng `feature_groups` để bỏ/giữ từng nhóm đặc trưng).
