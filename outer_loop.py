import numpy as np
from inner_loop import PythonRobotDynamics
from config import (
    MJM_METHOD, LEADER_COOLDOWN, BLEND_STEPS, PHI_ANGLE,
    N_CONFLICT_REQUIRED, N_CONFIDENCE_REQUIRED,
    K_H, K_H_REDUCE_FACTOR, K_H_REDUCE_STEPS, FORCE_DEADZONE
)


def compute_disagreement(f_h: np.ndarray, f_r: np.ndarray) -> float:
    norm_h = np.linalg.norm(f_h)
    norm_r = np.linalg.norm(f_r)

    if norm_r == 0:
        return 0.0
    if norm_h < FORCE_DEADZONE:
        return 1.0

    return float(np.dot(f_h, f_r) / (norm_h * norm_r))

class LocalPythonBridge:
    def __init__(self, dt=None):
        from config import DT
        self._connected = False
        self.dynamics = PythonRobotDynamics(dt=dt if dt is not None else DT)

    def connect(self, retries=5, delay=1.0):
        self._connected = True
        print("[Bridge] Connected to Local Python Dynamics Solver")
        return True

    def disconnect(self):
        self._connected = False

    def reset(self, x_init: np.ndarray = None):
        """Reset dynamics state về điểm xuất phát của scenario mới."""
        self.dynamics.reset(x_init)

    def step(self, x_ref: np.ndarray, x_h: np.ndarray, is_leader: float, k_h: float = None):
        if not self._connected:
            return x_ref.copy(), np.zeros(3), x_ref.copy(), np.zeros(3)
        return self.dynamics.step(x_ref, x_h, is_leader, k_h=k_h)


def run_control_loop(test_data, gmm_models, gmm_scaler,
                     svgp_model, svgp_scaler_x, svgp_scaler_y,
                     bridge: LocalPythonBridge, goals: dict, true_target_id: int):

    from shared_control_lib import (
        ControlMode, compute_goal_probabilities, predict_next_point_svgp,
        dynamic_minimum_jerk_trajectory, fitts_law_duration,
        paper_mjm_step,
        P_HIGH, P_HYSTERESIS, DT,
        SVGP_HISTORY_SIZE, GMM_WINDOW_SIZE, TAU_SOFTMAX
    )
    import time as _time

    n_steps = len(test_data)
    current_mode = ControlMode.FOLLOWER

    robot_trajectory    = []
    x_ref_history       = []
    x_d_history         = []
    modes               = []
    phi_history         = []
    f_h_history         = []
    f_r_history         = []
    assist_energy_history = []  # A = -f_h^T(x_d - x_robot)
    t_fitts_remaining_history = []
    max_probs_history   = []
    probability_history = {tid: [] for tid in gmm_models}
    # ⑥ Lịch sử chuyển đổi mode: {'step', 'time_s', 'from', 'to', 'goal_id'}
    mode_transitions    = []

    timing = {'gmm': 0., 'svgp': 0., 'mjm': 0., 'bridge': 0.,
              'gmm_n': 0, 'svgp_n': 0, 'mjm_n': 0, 'bridge_n': 0}

    # State Variables chung
    current_goal_id = None

    # ── CURRENT MODE: Replan mỗi lần vào LEADER ──────────────────────────────
    mjm_trajectory_cache = None
    mjm_step_index = 0
    mjm_cached_goal_id = None
    t_fitts_direct_initial = 0.0
    direct_step_counter = 0

    # ── PAPER MODE: Tracking τ trên đường cong gốc X0→Xf ─────────────────
    # x_0_paper : LUON = test_data[0] (diem bat dau quy dao goc), KHONG thay doi
    # tau_paper : τ hiện tại, được bisection tìm từ vi trí robot mỗi bước
    #             (tau_prev chỉ dùng làm khởi điểm tìm kiếm, không reset khi vào LEADER)
    x_0_paper = test_data[0].copy()  # Cố định từ đầu
    tau_paper = 0.0
    paper_cached_goal_id = None  # Phát hiện khi goal đổi → print log
    t_f_paper_current = 12.0    # t_f cho PAPER mode (cố định theo dataset)

    # Early-stop: điểm dừng sớm khi vận tốc < 5 cm/s cách phần còn lại
    # Được cắt sẵn từ test_data trước khi vào vòng lặp
    velocities_raw = np.linalg.norm(np.diff(test_data, axis=0), axis=1) / DT
    below_thresh = velocities_raw < 0.05  # 5 cm/s
    if not np.all(below_thresh):
        last_active = int(np.where(~below_thresh)[0][-1])
        n_keep = last_active + 2
        test_data = test_data[:n_keep]
        n_steps   = len(test_data)
    early_stop_pos = test_data[-1].copy()  # Điểm đích = điểm dừng sớm

    follower_cooldown_steps = 0
    consecutive_conflict_count = 0  # Bộ đếm xung đột liên tiếp
    consecutive_confidence_count = 0 # Bộ đếm xác suất tự tin liên tiếp
    candidate_leader_goal = None     # Goal tiềm năng đang được theo dõi
    follower_blend_steps_remaining = 0
    blend_anchor_pos = None  # Vi tri robot luc thoat LEADER (diem neo blend)
    leader_steps_count = 0   # So buoc da o LEADER (de giam dan K_H)
    # PAPER leader-entry blend: nội suy mượt khi FOLLOWER → LEADER
    leader_entry_blend_remaining = 0  # Số bước blend còn lại khi mới vào LEADER
    leader_entry_anchor = None        # Vị trí robot lúc bắt đầu vào LEADER

    print(f"  MJM_METHOD={MJM_METHOD}  |  P_HIGH={P_HIGH}")

    current_x_robot = test_data[0].copy()
    for t in range(n_steps):
        x_h = test_data[t].copy()

        # ── t=0: Khởi tạo ────────────────────────────────────────────────
        if t == 0:
            x_ref = test_data[0].copy()
            current_x_robot, f_r_raw, x_d_step, f_h = bridge.step(x_ref, x_h, 0.0)

            x_ref_history.append(x_ref.copy())
            x_d_history.append(x_d_step.copy())
            robot_trajectory.append(current_x_robot.copy())
            modes.append(ControlMode.FOLLOWER)
            # Tính phi ngay sau khi có lực của bước t=0
            phi = compute_disagreement(f_h, f_r_raw)
            phi_history.append(phi)
            f_h_history.append(f_h.copy())
            f_r_history.append(f_r_raw.copy())
            # Assist Energy: A = -f_h^T(x_d - x_robot)
            tracking_error_0 = x_d_step - current_x_robot
            assist_energy_history.append(float(-np.dot(f_h, tracking_error_0)))
            t_fitts_remaining_history.append(0.0)
            for tid in gmm_models:
                probability_history[tid].append(1.0 / len(gmm_models))
            max_probs_history.append(1.0 / len(gmm_models))
            continue

        t_fitts_remaining = 0.0
        # Lấy góc phi (đã được tính từ lực của bước trước) để ra quyết định cho bước này
        phi = phi_history[-1]

        if len(robot_trajectory) >= 2:
            v_current = (robot_trajectory[-1] - robot_trajectory[-2]) / DT
        else:
            v_current = np.zeros(3)

        if len(robot_trajectory) >= 3:
            v_prev    = (robot_trajectory[-2] - robot_trajectory[-3]) / DT
            a_current = (v_current - v_prev) / DT
        else:
            a_current = np.zeros(3)

        # GMM Classification
        points_history = test_data[0:t+1]
        _t0 = _time.perf_counter()
        probs = compute_goal_probabilities(
            points_history, gmm_models, gmm_scaler,
            window_size=GMM_WINDOW_SIZE, tau=TAU_SOFTMAX
        )
        timing['gmm'] += _time.perf_counter() - _t0
        timing['gmm_n'] += 1

        for tid in gmm_models:
            probability_history[tid].append(probs[tid])

        max_prob_goal = max(probs, key=probs.get)
        max_prob      = probs[max_prob_goal]
        max_probs_history.append(max_prob)

        # ROLE ARBITRATION
        prev_mode = modes[-1] if modes else ControlMode.FOLLOWER

        if prev_mode == ControlMode.LEADER:
            if phi < np.cos(np.deg2rad(PHI_ANGLE)):
                consecutive_conflict_count += 1
            else:
                consecutive_conflict_count = 0   # Reset khi phi OK trở lại
        else:
            consecutive_conflict_count = 0       # FOLLOWER không bị tính xung đột để phạt cooldown

        if consecutive_conflict_count >= N_CONFLICT_REQUIRED:
            follower_cooldown_steps = LEADER_COOLDOWN
            consecutive_conflict_count = 0   # Reset sau khi kích hoạt cooldown

        # Probability Hysteresis: Giảm ngưỡng để duy trì LEADER
        effective_threshold = P_HIGH
        if prev_mode == ControlMode.LEADER and max_prob_goal == current_goal_id:
            effective_threshold = P_HIGH - P_HYSTERESIS

        # Confidence Filter
        if max_prob >= effective_threshold:
            if max_prob_goal == candidate_leader_goal:
                consecutive_confidence_count += 1
            else:
                candidate_leader_goal = max_prob_goal
                consecutive_confidence_count = 1
        else:
            consecutive_confidence_count = 0
            candidate_leader_goal = None

        # Conflict Detection (Có Cooldown) → Bắt buộc FOLLOWER
        if follower_cooldown_steps > 0:
            current_mode = ControlMode.FOLLOWER
            mjm_trajectory_cache = None
            current_goal_id = None
            follower_cooldown_steps -= 1
            consecutive_confidence_count = 0 # Reset confidence khi đang bị phạt

        # High Confidence (Sustained) → LEADER
        elif consecutive_confidence_count >= N_CONFIDENCE_REQUIRED:

            just_entered_leader = (prev_mode != ControlMode.LEADER)
            current_mode = ControlMode.LEADER
            current_goal_id = candidate_leader_goal
            if current_goal_id == true_target_id:
                goal_pos = early_stop_pos.copy()  # Dung dich -> diem dung som
            else:
                goal_pos = goals[current_goal_id]  # Sai dich -> centroid mac dinh

            if MJM_METHOD == "PAPER":
                # ── PAPER MODE ───────────────────────────────────────────────────
                # x_0_paper = test_data[0]: cố định, không đổi
                # tau_paper: bisection tìm từ vị trí robot hiện tại, không reset
                goal_changed = (current_goal_id != paper_cached_goal_id)
                if goal_changed or just_entered_leader:
                    paper_cached_goal_id = current_goal_id
                    # Khởi động leader-entry blend (giảm gai vận tốc)
                    leader_entry_blend_remaining = BLEND_STEPS
                    leader_entry_anchor = current_x_robot.copy()
                    print(f"   [PAPER-MJM] t={t*DT:.2f}s, "
                          f"X_0={x_0_paper}, X_f={goal_pos}, "
                          f"dist={np.linalg.norm(goal_pos - x_0_paper):.4f}m, "
                          f"t_f={t_f_paper_current:.2f}s")

                # Mỗi bước: tìm τ từ vị trí hiện tại → tính điểm tiếp theo
                _t0 = _time.perf_counter()
                raw_mjm_point, _, tau_paper = paper_mjm_step(
                    x_current=current_x_robot,
                    x_0=x_0_paper,
                    x_f=goal_pos,
                    t_f=t_f_paper_current,
                    dt=DT,
                    tau_prev=tau_paper,
                )
                timing['mjm'] += _time.perf_counter() - _t0
                timing['mjm_n'] += 1

                if tau_paper >= 1.0:
                    tau_paper = 1.0  # Giữ nguyên ở 1.0 để robot duy trì tại đích

                t_fitts_remaining = (1.0 - tau_paper) * t_f_paper_current  # Thời gian còn lại ước tính

                current_mjm_point = raw_mjm_point

            else:
                # ── CURRENT MODE (Replan) ──────────────────────────────────
                goal_changed = (current_goal_id != mjm_cached_goal_id)
                if goal_changed or just_entered_leader:
                    x_0_mjm = current_x_robot.copy()
                    mjm_cached_goal_id = current_goal_id
                    t_remaining = (n_steps - 1 - t) * DT  # Thời gian thực tế còn lại
                    t_fitts = min(fitts_law_duration(x_0_mjm, goal_pos), t_remaining)
                    print(f"   [CURRENT-MJM] t={t*DT:.2f}s, "
                          f"goal={current_goal_id}, "
                          f"Fitts t_f={t_fitts:.2f}s")

                    _t0 = _time.perf_counter()
                    mjm_trajectory_cache = dynamic_minimum_jerk_trajectory(
                        x_0=x_0_mjm, v_0=0, x_f=goal_pos,
                        t_total=t_fitts, dt=DT, a_0=0
                    )
                    timing['mjm'] += _time.perf_counter() - _t0
                    timing['mjm_n'] += 1
                    mjm_step_index = 1 if len(mjm_trajectory_cache) > 1 else 0
                    t_fitts_direct_initial = t_fitts
                    direct_step_counter = 0

                if mjm_trajectory_cache is not None and mjm_step_index < len(mjm_trajectory_cache):
                    current_mjm_point = mjm_trajectory_cache[mjm_step_index]
                    mjm_step_index += 1
                else:
                    # Đã đến đích → giữ nguyên tại goal_pos, không dừng mô phỏng
                    current_mjm_point = goal_pos.copy()

                t_fitts_remaining = max(t_fitts_direct_initial - direct_step_counter * DT, 0.0)
                direct_step_counter += 1

            x_ref_new = current_mjm_point.copy()

        # Low Confidence → FOLLOWER
        else:
            current_mode = ControlMode.FOLLOWER
            mjm_trajectory_cache = None
            current_goal_id = None

        # LEADER → FOLLOWER: kích hoạt transition blend
        if current_mode == ControlMode.FOLLOWER and prev_mode == ControlMode.LEADER:
            follower_blend_steps_remaining = BLEND_STEPS
            blend_anchor_pos = current_x_robot.copy()

        # FOLLOWER: Dùng SVGP để dự đoán \hat{x}
        if current_mode == ControlMode.FOLLOWER:
            _t0 = _time.perf_counter()
            x_svgp = predict_next_point_svgp(
                points_history, svgp_scaler_x, svgp_scaler_y,
                svgp_model, history_size=SVGP_HISTORY_SIZE
            )
            timing['svgp'] += _time.perf_counter() - _t0
            timing['svgp_n'] += 1

            # Blend mượt từ vị trí robot (lúc thoát LEADER) sang SVGP
            if follower_blend_steps_remaining > 0 and blend_anchor_pos is not None:
                if BLEND_STEPS == 0:
                    alpha = 1.0
                else:
                    alpha = 1.0 - (follower_blend_steps_remaining-1) / BLEND_STEPS
                x_ref_new = (1.0 - alpha) * blend_anchor_pos + alpha * x_svgp
                follower_blend_steps_remaining -= 1
            else:
                x_ref_new = x_svgp.copy()

        # Ghi nhận chuyển đổi mode
        if current_mode != prev_mode:
            mode_transitions.append({
                'step':    t,
                'time_s':  round(t * DT, 3),
                'from':    prev_mode,
                'to':      current_mode,
                'goal_id': current_goal_id,
            })

        # Log khi chuyển sang LEADER
        if current_mode == ControlMode.LEADER and prev_mode != ControlMode.LEADER:
            print(f"   [+] {t*DT:5.2f}s: LEADER (Target {current_goal_id})")

        x_ref = x_ref_new.copy()
        is_leader = 1.0 if current_mode == ControlMode.LEADER else 0.0

        # Tinh K_H dong: giam dan trong LEADER, khoi phuc khi FOLLOWER
        if current_mode == ControlMode.LEADER:
            leader_steps_count += 1
            exponent = min(leader_steps_count, K_H_REDUCE_STEPS)
            k_h_current = K_H * (K_H_REDUCE_FACTOR ** exponent)
        else:
            leader_steps_count = 0
            k_h_current = K_H

        _t0 = _time.perf_counter()
        current_x_robot, f_r, x_d_step, f_h = bridge.step(x_ref, x_h, is_leader, k_h=k_h_current)
        timing['bridge'] += _time.perf_counter() - _t0
        timing['bridge_n'] += 1

        # f_r thô vẫn được log; phi dùng f_r_for_phi đã smooth (rolling window)
        f_r_smooth = f_r.copy()

        x_ref_history.append(x_ref.copy())
        x_d_history.append(x_d_step.copy())
        robot_trajectory.append(current_x_robot.copy())
        modes.append(current_mode)
        
        # Tính toán phi cho bước HIỆN TẠI dựa trên lực HIỆN TẠI vừa sinh ra
        current_phi = compute_disagreement(f_h, f_r_smooth)
        phi_history.append(current_phi)
        
        # Assist Energy: A = -f_h^T(x_d - x_robot)
        tracking_error = x_d_step - current_x_robot
        assist_energy_history.append(float(-np.dot(f_h, tracking_error)))
        t_fitts_remaining_history.append(t_fitts_remaining)
        
        f_h_history.append(f_h.copy())
        f_r_history.append(f_r_smooth.copy())


    def _avg(key):
        n = timing[key + '_n']
        return timing[key] / n if n > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"  TIMING SUMMARY")
    print(f"  GMM:    {timing['gmm']:.2f}s  ({_avg('gmm')*1000:.1f} ms/call)")
    print(f"  SVGP:   {timing['svgp']:.2f}s  ({_avg('svgp')*1000:.1f} ms/call)")
    print(f"  MJM:    {timing['mjm']:.2f}s  ({_avg('mjm')*1000:.1f} ms/call, {timing['mjm_n']} calls)")
    print(f"  BRIDGE: {timing['bridge']:.2f}s  ({_avg('bridge')*1000:.1f} ms/call)")
    print(f"{'='*50}\n")

    # ⑥ Thêm thống kê chuyển đổi mode vào summary
    n_leader_entries  = sum(1 for tr in mode_transitions if tr['to'] == ControlMode.LEADER)
    n_leader_exits    = sum(1 for tr in mode_transitions if tr['from'] == ControlMode.LEADER)
    print(f"  Mode switches: {len(mode_transitions)} total "
          f"| LEADER entries: {n_leader_entries} | exits: {n_leader_exits}")

    n_len = len(x_ref_history)
    for tid in gmm_models:
        probability_history[tid] = probability_history[tid][:n_len]

    return {
        'robot_trajectory': np.array(robot_trajectory),
        'x_ref_history':    np.array(x_ref_history),
        'x_d_history':      np.array(x_d_history),
        'modes':            modes,
        'probabilities':    probability_history,
        'max_probs':        np.array(max_probs_history[:n_len]),
        'phi_history':      np.array(phi_history),
        'f_h_history':      np.array(f_h_history),
        'f_r_history':      np.array(f_r_history),
        'assist_energy_history': np.array(assist_energy_history),
        't_fitts_remaining_history': np.array(t_fitts_remaining_history),
        'mode_transitions': mode_transitions,
        'n_steps_completed': len(x_ref_history),  # Số bước thực tế (có thể < n_steps nếu dừng sớm)
    }