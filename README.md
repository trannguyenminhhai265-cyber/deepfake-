# Deepfake Detection — `deepfake_pj_group.ipynb`

Pipeline phát hiện ảnh deepfake dùng đặc trưng cổ điển + mô hình sklearn, trên dataset `project_data` (20.000 ảnh real + 20.000 ảnh fake).

## Luồng xử lý

```
ảnh → resize 256×256 (xám) → ảnh dư PCA (bỏ 32 thành phần chính)
    → FFT (64) + LBP (59) + Noise (64) → feature vector 187 chiều
    → StandardScaler → split 80/10/10 → SVM / RF / GBM
```

## Phân công

| Người | Nhiệm vụ | Cell trong notebook |
|---|---|---|
| **A — Data** | Lấy `project_data`, giải nén, resize, chia train/val/test | Setup + `[A]` |
| **B — Features** | FFT + LBP + Noise → ghép feature vector + nhóm đặc trưng cho ablation | `[B]` |
| **C — Model** | Train SVM/RF/GBM, đánh giá AUC/F1/Confusion, ablation, vẽ biểu đồ | `[C]` |

## Những gì đã sửa ở phần Features (người B)

1. **Sửa lỗi LBP.** Bản cũ dùng `method='uniform'` (chỉ sinh ~10 mẫu) nhưng lại đặt `LBP_BINS=59`, khiến **49/59 chiều luôn bằng 0** (đặc trưng chết). Đã đổi sang `method='nri_uniform'` (đúng 59 mẫu) → dùng đủ 59 bins.
2. **`feature_groups` để ablation đúng.** Lưu vị trí cột của từng nhóm `{'fft':(0,64), 'lbp':(64,123), 'noise':(123,187)}` vào mỗi file `.pkl`, thay cho việc hardcode chỉ số slice (dễ sai). Người C dùng nhóm này để bỏ/giữ từng đặc trưng **trên cả tập test**, không chỉ train.
3. **Chuẩn hóa không rò rỉ dữ liệu.** `StandardScaler` chỉ `fit` trên train rồi `transform` val/test.
4. **`FEATURE_DIM` tính tự động** từ số bins (không hardcode 187) + `assert` kiểm tra chiều vector.
5. **Lấy dữ liệu bền hơn:** thử Drive đã mount trước, không thấy thì tải bằng `gdown`.

## Phần Model (người C) — đã thêm sẵn để chạy

- Nạp `train/val/test.pkl`, train **SVM (RBF)**, **Random Forest**, **Gradient Boosting**.
- Đánh giá **Accuracy / F1 / AUC / Confusion matrix** trên val và test → bảng so sánh mô hình.
- **Ablation study**: `full / no_fft / no_lbp / no_noise / only_fft / only_lbp / only_noise` → bảng AUC + `delta_auc_vs_full`.
- Vẽ **ROC**, **Confusion matrix**, biểu đồ **ablation**.
- `SVM_MAX_TRAIN` (mặc định 8000) để giới hạn mẫu train cho RBF-SVM cho nhanh; đặt `None` để dùng toàn bộ.

## File bàn giao B → C

`train.pkl`, `val.pkl`, `test.pkl` (mỗi file gồm `X`, `y`, `paths`, `feature_names`, `feature_groups`, `config`) và `scaler.pkl`.
