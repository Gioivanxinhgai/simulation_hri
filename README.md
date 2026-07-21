# pHRI Shared Control Simulation

> **Mô phỏng hệ thống Điều khiển Chia sẻ trong Tương tác Người–Robot (physical Human-Robot Interaction)**
>
> Mô hình sử dụng GMM để nhận dạng ý định, SVGP để dự đoán quỹ đạo, và MJM để tạo quỹ đạo robot, tích hợp trong vòng lặp điều khiển hai tầng (Outer/Inner Loop).

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Luồng dữ liệu (Data Flow)](#2-luồng-dữ-liệu-data-flow)
3. [Mô tả các file chính](#3-mô-tả-các-file-chính)
4. [Tham số cấu hình (config.py)](#4-tham-số-cấu-hình-configpy)
5. [Cơ chế chuyển đổi chế độ (Role Arbitration)](#5-cơ-chế-chuyển-đổi-chế-độ-role-arbitration)
6. [Bộ dữ liệu và kịch bản thử nghiệm](#6-bộ-dữ-liệu-và-kịch-bản-thử-nghiệm)
7. [Đầu ra và chỉ số đánh giá](#7-đầu-ra-và-chỉ-số-đánh-giá)
8. [Hướng dẫn chạy](#8-hướng-dẫn-chạy)

---

## 1. Tổng quan kiến trúc

Hệ thống được tổ chức theo kiến trúc **hai vòng lặp lồng nhau** (nested dual-loop) theo sơ đồ kiến trúc chính thức:

```
 F_h(t-1) ┐
 F_r(t-1) ┘
     │
┌────▼────────────────────────────────────────────────────────────────────┐
│  OUTER LOOP                                                             │
│                                                                          │
│  ┌──────────────────────────────────────┐   ┌───────────────────────┐  │
│  │         ROLE ARBITRATION             │   │ REFERENCE MOTION GEN. │  │
│  │                                      │   │                       │  │
│  │  [Disagreement Detection]            │   │  ┌─────────────────┐  │  │
│  │        │ Conflict?                   │   │  │    FOLLOWER     │  │  │
│  │   True ▼       False ▼              │   │  │ ┌─────────────┐ │  │  │
│  │  (→FOLLOWER) [Goal Classification]  │   │  │ │ Admittance  │ │  │  │
│  │              X_h,t-5:t (GMM)        │   │  │ │  Control    │◀┼──┼──┼── F_h(t)
│  │                │ Confidence?         │   │  │ └──────┬──────┘ │  │  │
│  │          Low ▼       High ▼         │──▶│  │  x̂_h,ẋ_h      │  │  │
│  │        (FOLLOWER)  Predicted Goal   │   │  │ ┌─────────────┐ │  │  │
│  │                                      │   │  │ │    SVGP     │ │  │  │
│  └──────────────────────────────────────┘   │  │ │ X_h,t-10:t │ │  │  │
│                                              │  │ └─────────────┘ │  │  │
│                                              │  └─────────────────┘  │  │
│                                              │  ┌─────────────────┐  │  │
│                                              │  │     LEADER      │  │  │
│                                              │  │  x(t), t_f     │  │  │
│                                              │  │ Minimum Jerk   │  │  │
│                                              │  │    Model       │  │  │
│                                              │  └─────────────────┘  │  │
│                                              └──────────┬────────────┘  │
└─────────────────────────────────────────────────────────┼───────────────┘
                                           x_d, ẋ_d      │
                       ┌─────────────────────────────────▼───────────┐
                       │                  INNER LOOP                  │
                       │                                              │
                       │           ┌──────────────┐                  │
                       │           │ PD Controller│──▶ F_r(t) ──▶ 🤖│
                       │           └──────────────┘          │      │
                       │                   ▲                  │      │
                       │           x_r, ẋ_r (feedback)        │      │
                       └───────────────────────────────────────┘      │
                                                       F_h(t) ◀── 👋
```

### Các chế độ điều khiển

| Chế độ | Mô tả | Khối tạo tín hiệu `x_d, ẋ_d` |
|---|---|---|
| **FOLLOWER** | Robot đi theo ý định người dự báo bởi SVGP + Admittance | Admittance Control + SVGP (cửa sổ 10 điểm) |
| **LEADER** | Robot chủ động dẫn hướng đến Predicted Goal | Minimum Jerk Model với `x(t)`, `t_f` |

---

## 2. Luồng dữ liệu (Data Flow)

```
                F_h(t-1), F_r(t-1)  ◀────────────────────────────┐
                       │                                           │
         ┌─────────────▼──────────────┐                           │
         │     ROLE ARBITRATION       │                           │
         │  [Disagreement Detection]  │                           │
         │     φ = f_h · f_r /        │                           │
         │       (|f_h| |f_r|)        │                           │
         │           │                │                           │
         │    Conflict? (φ < 0)       │                           │
         │   True ▼       False ▼    │                           │
         │ (FOLLOWER   [Goal Classif.]│                           │
         │  cooldown)  GMM(X_h,t-5:t)│                           │
         │             → P(goal)      │                           │
         │             Confidence?    │                           │
         │         Low ▼  High ▼     │                           │
         │     (FOLLOWER) LEADER mode │                           │
         └──────┬─────────────┬───────┘                          │
                │             │                                   │
          FOLLOWER         LEADER                                 │
                │             │                                   │
    ┌───────────▼──┐   ┌──────▼────────────────┐                 │
    │  FOLLOWER    │   │  LEADER               │                 │
    │  ┌─────────┐ │   │  Minimum Jerk Model   │                 │
    │  │Adm. Ctrl│◀┼───┼── F_h(t)              │                 │
    │  │(x̂_h,ẋ_h│ │   │  Input: x(t), t_f     │                 │
    │  └────┬────┘ │   │  → x_d, ẋ_d          │                 │
    │       ↕      │   └──────────────────────-┘                 │
    │  ┌─────────┐ │                                              │
    │  │  SVGP   │ │                                              │
    │  │X_h,t-10:t│                                              │
    │  └────┬────┘ │                                              │
    └───────┼───────┘                                             │
            │ x_d, ẋ_d                                            │
            ▼                                                      │
  ┌──────────────────────┐                                        │
  │   PD Controller      │  f_r = Kp*(x_d - x_r) + Kd*(0 - v_r)│
  │   (INNER LOOP)       │─────────────────────────▶ F_r(t) ─────┘
  └──────────┬───────────┘                              │
             │ F_r(t) → Robot                           │
             ▼                                          │
  ┌──────────────────────┐                              │
  │  Robot Dynamics      │  ◀── F_h(t) = K_H*(x_h - x_r)
  │  (Forward Dynamics)  │                              │
  │  + Friction Model    │                              │
  └──────────┬───────────┘                              │
             │ x_r, ẋ_r  ─────────────────────────────▶┘ (feedback)
             │
             ▼
  ┌──────────────────────┐
  │ HRI Metrics (logged) │  θ, Assist Index, Assist Energy, Phi
  └──────────────────────┘
```

**Các biến trạng thái qua từng bước thời gian:**

| Ký hiệu | Vị trí | Ý nghĩa |
|---|---|---|
| `X_h, t-5:t` | Outer – GMM | Cửa sổ 5 điểm vị trí người → Goal Classification |
| `X_h, t-10:t` | Outer – SVGP | Cửa sổ 10 điểm lịch sử → dự đoán quỹ đạo FOLLOWER |
| `F_h(t-1), F_r(t-1)` | Outer – Role Arbitration | Lực bước trước → Disagreement Detection |
| `x̂_h, ẋ_h` | Outer – Admittance | Vị trí/vận tốc nhận cảm qua Admittance Control |
| `x_d, ẋ_d` | Outer → Inner | Điểm đích + vận tốc đặt truyền xuống Inner Loop |
| `F_r(t)` | Inner Loop | Lực PD robot tác động = `Kp*(x_d - x_r) + Kd*(0 - v_r)` |
| `F_h(t)` | Inner Loop | Lực người tác động = `K_H * (x_h - x_robot)` |
| `x_r, ẋ_r` | Inner Loop | Vị trí & vận tốc thực tế của robot (feedback) |
| `φ (phi)` | Outer – Disagreement | Chỉ số đồng thuận = `cos(θ)` giữa `F_h` và `F_r` |

---

## 3. Mô tả các file chính

### `run_simulation.py` — Điểm vào chính
- Load GMM và SVGP models từ checkpoint.
- Duyệt qua tất cả các file test CSV.
- Gọi `run_control_loop()` cho từng trajectory.
- Tính và lưu metrics (MAE, RMSE, Mean Theta, Mean Assist Index).
- Xuất file CSV chi tiết và hình ảnh đồ thị.
- In bảng summary tổng hợp ra file `summary.txt`.

### `outer_loop.py` — Vòng lặp ngoài (Nhận dạng, Quyết định & Reference Motion)
**Class `LocalPythonBridge`**: Cầu nối giữa Outer Loop và Inner Loop dynamics.

**Function `run_control_loop()`**: Vòng lặp điều khiển chính, thực hiện toàn bộ Outer Loop:

**① Disagreement Detection** (cổng đầu tiên của Role Arbitration):
- Dùng `F_h(t-1)` và `F_r(t-1)` từ bước trước để tính φ = cos(θ).
- Nếu `φ < cos(PHI_ANGLE)` liên tiếp `N_CONFLICT_REQUIRED` bước → phát hiện **Conflict** → bắt buộc FOLLOWER trong `LEADER_COOLDOWN` bước.

**② Goal Classification** (chỉ chạy khi không Conflict):
- **GMM**: Tính `P(goal | X_h, t-5:t)` trên cửa sổ trượt 5 điểm lịch sử vị trí người.
- Nếu `max_prob ≥ P_HIGH` liên tiếp `N_CONFIDENCE_REQUIRED` bước → **High Confidence** → kích hoạt LEADER.

**③ Reference Motion Generator — FOLLOWER block**:
- **Admittance Control**: Nhận `F_h(t)` thực tế, tính `x̂_h, ẋ_h` (vị trí người qua bộ lọc).
- **SVGP**: Dự đoán điểm tiếp theo từ `X_h, t-10:t` (cửa sổ 10 điểm).
- Blending mượt `BLEND_STEPS` bước khi chuyển đổi chế độ.

**④ Reference Motion Generator — LEADER block**:
- **MJM (CURRENT mode)**: Replan toàn bộ quỹ đạo mỗi lần mới vào LEADER từ `x(t)` đến `goal` trong thời gian `t_f`.
- **MJM (PAPER mode)**: Track tham số τ liên tục trên đường cong cố định `X_0 → X_f` theo phương trình bài báo.
- `t_f` được ước lượng bằng **Fitts' Law** hoặc **GT Time Cheat** (`USE_GT_TIME_CHEAT = True`).
- **K_H động**: Giảm dần độ cứng tay người theo `K_H_REDUCE_FACTOR` khi ở LEADER.

### `inner_loop.py` — Vòng lặp trong (Động học cấp thấp)
**Class `PythonRobotDynamics`**: Chỉ thực hiện **PD Control + Robot Dynamics** với sub-stepping (`N_SUB = 100`).

> [!IMPORTANT]
> Theo kiến trúc chính thức, **Admittance Control thuộc Outer Loop (FOLLOWER block)**, không thuộc Inner Loop. Inner Loop chỉ nhận `x_d, ẋ_d` đã được tính sẵn.

Các bước trong Inner Loop mỗi `DT`:
1. **Lực người**: `F_h = K_H * (x_h - x_robot)` — mô hình hóa tay người như lò xo.
2. **PD Controller**: `F_r = Kp * (x_d - x_r) + Kd * (0 - v_r)` — bám theo `x_d` từ Outer Loop.
3. **Ma sát (Friction)**: `F_f = -tanh(100·v) * (c_v·|v| + F_c)`
4. **Forward Dynamics**: `ẍ_r = (F_r + F_h_eff + F_f) / M_rob`
   - `F_h_eff = (1 - is_leader) * F_h` (giảm ảnh hưởng vật lý của người khi ở LEADER)
5. Tích phân Euler để cập nhật `x_r, ẋ_r`.

### `shared_control_lib.py` — Thư viện dùng chung
Chứa tất cả các hàm tiện ích, model loading, và plotting:

| Hàm/Class | Chức năng |
|---|---|
| `ControlMode` | Enum: `FOLLOWER`, `LEADER` |
| `load_gmm_system()` | Load GMM models + scaler từ `.pkl` |
| `load_svgp_system()` | Load SVGP model + scaler từ checkpoint dir |
| `read_and_smooth_3d()` | Đọc CSV, làm mịn bằng Gaussian Filter (sigma=0.2) |
| `compute_goal_probabilities()` | GMM log-likelihood → softmax → P(goal) |
| `predict_next_point_svgp()` | SVGP dự đoán điểm tiếp theo từ cửa sổ lịch sử |
| `dynamic_minimum_jerk_trajectory()` | Tạo quỹ đạo MJM đầy đủ (CURRENT mode) |
| `paper_mjm_step()` | Một bước MJM theo phương trình bài báo (PAPER mode) |
| `fitts_law_duration()` | Tính thời gian di chuyển theo Fitts' Law |
| `calculate_and_plot_hri_metrics()` | Tính và vẽ Theta, Assist Index, Assist Energy |
| `plot_simulation_results()` | Vẽ Goal Probability + Forces + Velocity |
| `plot_xd_vs_xr()` | Vẽ so sánh quỹ đạo mong muốn vs thực tế |
| `plot_phi()` | Vẽ Disagreement Index theo thời gian |

### `config.py` — Tham số tập trung

Toàn bộ tham số hệ thống được định nghĩa tập trung tại đây. Xem chi tiết ở mục 4.

---

## 4. Tham số cấu hình (`config.py`)

### Đường dẫn & Dữ liệu

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `GMM_CHECKPOINT_DIR` | `"gmm_model_hri"` | Thư mục chứa GMM model |
| `SVGP_CHECKPOINT_DIR` | `"svgp_hri_m52"` | Thư mục chứa SVGP checkpoint |
| `TEST_FOLDER` | `"Experiment_Test_Trajectory_HRI"` | Thư mục chứa file CSV test |
| `SAVE_DIR_OVERRIDE` | `"Simulation_Results"` | Thư mục lưu kết quả đầu ra |

### Mô hình GMM (Intent Recognition)

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `GMM_WINDOW_SIZE` | `5` | Số điểm trong cửa sổ trượt tính log-likelihood |
| `TAU_SOFTMAX` | `8` | Nhiệt độ softmax (càng cao → phân phối mềm hơn) |
| `SIGMA` | `0.2` | Sigma Gaussian filter làm mịn trajectory đầu vào |

### Mô hình SVGP (Trajectory Prediction)

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `SVGP_HISTORY_SIZE` | `10` | Số điểm lịch sử dùng làm input cho SVGP |

### Outer Loop — Role Arbitration

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `P_LOW` | `0.6` | Ngưỡng xác suất thấp (không dùng trực tiếp) |
| `P_HIGH` | `0.8` | Ngưỡng xác suất để xét vào LEADER |
| `P_HYSTERESIS` | `0.05` | Giảm ngưỡng khi đang LEADER → duy trì ổn định |
| `N_CONFIDENCE_REQUIRED` | `5` | Số bước liên tiếp ≥ P_HIGH để vào LEADER |
| `N_CONFLICT_REQUIRED` | `5` | Số bước xung đột liên tiếp để kích hoạt cooldown |
| `LEADER_COOLDOWN` | `10` | Số bước bắt buộc FOLLOWER sau khi phát hiện xung đột |
| `BLEND_STEPS` | `10` | Số bước nội suy mượt khi chuyển đổi chế độ |
| `PHI_ANGLE` | `90°` | Góc ngưỡng phát hiện xung đột (φ < cos(90°) = 0) |
| `FORCE_DEADZONE` | `0.05 N` | Ngưỡng lực người tối thiểu để tính disagreement |

### Fitts' Law (Ước lượng thời gian MJM)

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `FITTS_A` | `4.3939` | Hằng số a trong `T = a + b*log2(2D/W)` |
| `FITTS_B` | `0.5224` | Hằng số b |
| `FITTS_W` | `0.08 m` | Độ rộng mục tiêu W |

### Thời gian & Sub-stepping

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `DT` | `12/191 ≈ 0.0628 s` | Time-step chính (~15.9 Hz) |
| `N_SUB` | `100` | Số sub-step trong Inner Loop mỗi bước DT |
| `MJM_METHOD` | `"CURRENT"` | Chế độ MJM: `"CURRENT"` (replan) hoặc `"PAPER"` |
| `USE_GT_TIME_CHEAT` | `True` | Dùng thời gian ground truth thay Fitts' Law |

### Inner Loop — Robot Dynamics

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `M_ROB` | `1.0 kg` | Khối lượng robot |
| `C_V` | `0.75` | Hệ số cản nhớt |
| `F_C` | `3 N` | Lực ma sát Coulomb |
| `K_P` | `1,000,000` | Gain tỉ lệ PD |
| `K_D` | `2*0.7*√K_P` | Gain vi phân PD (critically damped) |

### Admittance Control

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `M_ADM` | `1.0` | Khối lượng ảo Admittance |
| `K_ADM` | `100.0` | Độ cứng Admittance |
| `D_ADM` | `20.0` | Cản Admittance |

### Human Model

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `K_H` | `150 N/m` | Độ cứng tay người mặc định |
| `K_H_REDUCE_FACTOR` | `0.5` | Hệ số giảm K_H mỗi bước khi ở LEADER |
| `K_H_REDUCE_STEPS` | `6` | Số bước tối đa giảm K_H |

### Mục tiêu (Goals) — 3D Cartesian

| Target | Tọa độ [X, Y, Z] (m) |
|---|---|
| Target 1 | `[0.3796, 0.9779, 0.0169]` |
| Target 2 | `[0.0409, 1.3684, 0.0052]` |
| Target 3 | `[-0.0784, 1.0867, 0.0256]` |

---

## 5. Cơ chế chuyển đổi chế độ (Role Arbitration)

```
  F_h(t-1), F_r(t-1)
         │
         ▼
  ┌──────────────────────┐
  │ Disagreement Detect. │  φ = (F_h · F_r) / (|F_h| |F_r|)
  └──────────┬───────────┘
             │
      Conflict? (φ < cos(PHI_ANGLE), N_CONFLICT lần liên tiếp)
      ┌───── YES ──────────────────────────────────────────────┐
      │                                                    ▼
      │ NO                                   ┌────────────────────────┐
      │                                       │  FORCED FOLLOWER       │
      ▼                                       │  (Cooldown LEADER_     │
  ┌──────────────────────┐                   │   COOLDOWN bước)       │
  │  Goal Classification │                   └────────────────────────┘
  │  GMM(X_h, t-5:t)     │
  │  → P(goal_1/2/3)     │
  └──────────┬───────────┘
             │
      Confidence? (max_prob ≥ P_HIGH*, N_CONFIDENCE lần liên tiếp)
      ┌───── LOW ────────────────────────────────────────────────┐
      │                                                    │
      │ HIGH                                               ▼
      ▼                                   ┌────────────────────────────┐
  ┌──────────────────────────────┐        │        FOLLOWER            │
  │           LEADER             │        │  Admittance + SVGP         │
  │  MJM(x(t), t_f → goal)      │        │  x_d = Admittance(F_h)    │
  │  x_d, ẋ_d → Inner Loop      │        │      + SVGP(X_h,t-10:t)  │
  └──────────────────────────────┘        └────────────────────────────┘
```

> **(*)  Hysteresis**: Khi đang LEADER và goal không đổi, ngưỡng hiệu lực giảm xuống `P_HIGH - P_HYSTERESIS = 0.75` để tránh dao động chế độ liên tục.

**Điểm mấu chốt của thiết kế:**
- **Disagreement Detection là cổng ưu tiên cao nhất**: Conflict → ngay lập tức FOLLOWER, không cần qua Goal Classification.
- **FOLLOWER** dùng cả Admittance Control (phản hồi lực tức thì) lẫn SVGP (dự báo quỹ đạo).
- **LEADER** chỉ dùng Minimum Jerk Model, `F_h` bị giảm ảnh hưởng vật lý trong Inner Loop.

---

## 6. Bộ dữ liệu và kịch bản thử nghiệm

Dữ liệu test lưu trong thư mục `Experiment_Test_Trajectory_HRI/`, được liệt kê trong `test_file_list.json`.

### Mapping ScenarioId → Target

```
Target 1: ScenarioId = 1, 4, 9, 11, 15, 17
Target 2: ScenarioId = 2, 5, 7, 12, 13, 18
Target 3: ScenarioId = 3, 6, 8, 10, 14, 16
```

### Loại kịch bản (18 kịch bản)

| Loại | ScenarioId | Mô tả |
|---|---|---|
| Free Initial | 1, 2, 3 | Xuất phát tự do, đến mục tiêu tương ứng |
| Obstacle Initial | 4, 5, 6 | Xuất phát có vật cản |
| Change Initial | 7–12 | Thay đổi điểm xuất phát (lệch so với target) |
| Change + Obstacle | 13–18 | Kết hợp thay đổi điểm xuất phát và vật cản |

---

## 7. Đầu ra và chỉ số đánh giá

### Thư mục kết quả: `Simulation_Results/`

| File/Thư mục | Nội dung |
|---|---|
| `csv_logs/<traj>.csv` | Log chi tiết từng bước: time, mode, probs, x_h, x_ref, x_d, x_r, F_h, F_r, theta_deg |
| `prob_<traj>.png` | Xác suất mục tiêu theo thời gian |
| `forces_<traj>.png` | Lực người vs lực robot theo 3 trục |
| `velocity_<traj>.png` | Vận tốc ground truth vs prediction |
| `phi_<traj>.png` | Disagreement Index (Phi/Theta) |
| `xref_vs_gt_<traj>.png` | x_ref (outer loop) vs Human ground truth |
| `xd_vs_xr_<traj>.png` | x_d (PD desired) vs x_robot (actual) |
| `hri_metrics_<traj>.png` | (a) Theta, (b) Assist Index, (c) Assist Energy |
| `summary.txt` | Bảng tổng hợp MAE, RMSE, Mean Theta, Mean Assist |

### Chỉ số HRI

| Chỉ số | Công thức | Ý nghĩa |
|---|---|---|
| **MAE** | `mean(‖x_ref - x_h‖)` | Sai số vị trí trung bình |
| **Theta (θ)** | `arccos(f_h · f_r / (‖f_h‖ ‖f_r‖))` | Góc xung đột giữa lực người và robot |
| **Assist Index** | `f_r · f_h / ‖f_h‖` | Mức độ robot hỗ trợ hướng người đang đẩy |
| **Assist Energy** | `A = -f_h^T (x_d - x_robot)` | Năng lượng hỗ trợ (A<0: hỗ trợ, A>0: cản trở) |
| **Phi (φ)** | `f_h · f_r / (‖f_h‖ ‖f_r‖)` | Chỉ số đồng thuận (=1: đồng hướng, =-1: đối nghịch) |

---

## 8. Hướng dẫn chạy

### Yêu cầu

```bash
pip install numpy pandas matplotlib scipy gpflow
```

### Cấu trúc thư mục

```
d:\LAB\
├── config.py                          # Tham số tập trung
├── run_simulation.py                  # Entry point
├── outer_loop.py                      # Outer control loop
├── inner_loop.py                      # Inner robot dynamics
├── shared_control_lib.py              # Thư viện dùng chung
├── gmm_model_hri/
│   ├── gmm_models.pkl
│   └── scaler.pkl
├── svgp_hri_m52/
│   ├── svgp_model.pkl
│   ├── scaler_x.pkl
│   └── scaler_y.pkl
├── Experiment_Test_Trajectory_HRI/
│   ├── test_file_list.json
│   └── *.csv
└── Simulation_Results/               # Tự tạo khi chạy
    ├── csv_logs/
    └── *.png, summary.txt
```

### Chạy mô phỏng

```bash
cd d:\LAB
python run_simulation.py
```

### Thay đổi chế độ MJM

Trong `config.py`:
```python
MJM_METHOD = "CURRENT"   # Replan mỗi lần vào LEADER
# hoặc
MJM_METHOD = "PAPER"     # Track τ liên tục theo phương trình bài báo
```

### Tắt GT Time Cheat

```python
USE_GT_TIME_CHEAT = False   # Dùng Fitts' Law thay vì GT time
```
