function matlab_server() PORT = 25000;
DT = 12 / 189;
MAX_STEPS = 100000;
% Tham số của Admittance Control M_adm = diag([ 1, 1, 1 ]);
D_adm = diag([ 25, 25, 25 ]);
K_adm = diag([ 200, 200, 200 ]);
% Tham số của PD Controller K_p = diag([ 500, 500, 500 ]);
K_d = diag([ 45, 45, 45 ]);
% Động lực học của ROBOT M_rob = 1.0;
c_v = 0.5;
F_c = 2.0;
% % ── LEADER DECAY
    : tốc độ x_adm về 0 khi chuyển sang LEADER ────── LEADER_DECAY = 0.05;
% Mỗi sub - step, x_adm *= LEADER_DECAY

                           % Trạng thái của ROBOT x_robot = zeros(3, 1);
% Vị trí dx_robot = zeros(3, 1);
% Vận tốc x_adm = zeros(3, 1);
dx_adm = zeros(3, 1);
x_ref_prev = zeros(3, 1);

initialized = false;
% Khởi tạo server server_socket =
    tcpserver("0.0.0.0", PORT, "Timeout", 60, "ByteOrder", "little-endian");
while
  server_socket.NumBytesAvailable == 0 pause(0.05);
break end fprintf('[Server] Python connected.\n\n');

% % ── VÒNG LẶP CHÍNH ───────────────────────────────────────────────── step =
    0;
log_x = zeros(MAX_STEPS, 3);
log_xref = zeros(MAX_STEPS, 3);
log_fr = zeros(MAX_STEPS, 3);
log_fh = zeros(MAX_STEPS, 3);
log_phi = zeros(MAX_STEPS, 1);

while
  step < MAX_STEPS bytes_needed = 56;
wait_count = 0;
while
  server_socket.NumBytesAvailable < bytes_needed pause(0.001);
wait_count = wait_count + 1;
if wait_count
  > 120000 fprintf('[Server] Timeout waiting for data. Stopping.\n');
break;
end end

    if server_socket.NumBytesAvailable < bytes_needed;
break;
end

    % Đọc dữ liệu raw = read(server_socket, 7, "double");
x_ref_in = raw(1 : 3)'; f_h = raw(4 : 6)'; is_leader = raw(7);

if any (isnan(x_ref_in))
  ;
break;
end

    % % ── KHỞI TẠO / RESET KHI CHUYỂN KỊCH BẢN ──────────────────────── %
    Python chạy nhiều file test liên tiếp qua cùng 1 kết nối MATLAB.%
    Khi chuyển file,
    x_ref nhảy đột ngột(> 0.1m) → cần reset trạng
        % thái robot về vị trí đầu file mới để PD không bị sốc lực.needs_reset =
        ~initialized || norm(x_ref_in - x_ref_prev) > 0.1;

if needs_reset
  x_robot = x_ref_in;
x_adm = zeros(3, 1);
dx_adm = zeros(3, 1);
dx_robot = zeros(3, 1);
x_ref_prev = x_ref_in;
initialized = true;
end

    step = step + 1;

% % ── KỸ THUẬT SUB - STEPPING ─────────────────────────────────────── N_sub =
    100;
dt_sub = DT / N_sub;

        for
          sub = 1 : N_sub alpha_interp = sub / N_sub;
        x_ref_current = x_ref_prev + alpha_interp * (x_ref_in - x_ref_prev);

        if is_leader
          == 1.0 % % ── LEADER : Tắt Admittance,
              MJM trực tiếp điều khiển ──── %
                  Decay nhanh x_adm về 0 để không làm lệch x_d x_adm =
                  x_adm * LEADER_DECAY;
        dx_adm = dx_adm * LEADER_DECAY;

        x_d = x_ref_current;
        % MJM truyền thẳng, không qua Admittance dx_d = zeros(3, 1);
        else % % ── FOLLOWER : Admittance Control bật,
            người dẫn dắt ───── ddx_adm =
                M_adm \ (f_h - D_adm * dx_adm - K_adm * x_adm);
        dx_adm = dx_adm + ddx_adm * dt_sub;
        x_adm = x_adm + dx_adm * dt_sub;

        x_d = x_ref_current + x_adm;
        dx_d = dx_adm;
        end

            % % Tính f_r(PD Tracking) e_pos = x_d - x_robot;
        e_vel = dx_d - dx_robot;
        f_r = K_p * e_pos + K_d * e_vel;

        % % Friction F_f = -tanh(100 * dx_robot).*(c_v * abs(dx_robot) + F_c);

        % % Forward Dynamics ddx_robot = (f_r + f_h + F_f) / M_rob;
        dx_robot = dx_robot + ddx_robot * dt_sub;
        x_robot = x_robot + dx_robot * dt_sub;
        end

                % % ── Cập nhật x_ref_prev cho macro -
            step tiếp theo ──────────────── x_ref_prev = x_ref_in;

        % % ── Gửi kết quả về
                Python ─────────────────────────────────────── response =
            [x_robot( :); f_r( :)];
        write(server_socket, response, "double");

        % % ── Log ───────────────────────────────────────────────────────── log_x(
              step, :) = x_robot'; log_xref(step, :) =
            x_ref_in'; log_fr(step, :) = f_r'; log_fh(step, :) = f_h';

            norm_h = norm(f_h);
        norm_r = norm(f_r);
        % Con người ko tác động lực thì Disagreement =
            1 if norm_h < 1e-4 && norm_r >= 1e-4 log_phi(step) = 1.0;
        % Robot không tác động lực thì Disagreement =
            0 elseif norm_r < 1e-4 log_phi(step) = 0.0;
        % Tính Cos giữa f_h và f_r else log_phi(step) = dot(f_h, f_r) /
                                                        (norm_h * norm_r);
        end end %
            Kết thúc fprintf('\n[Server] Simulation done. Total steps: %d\n',
                             step);
        clear server_socket;
        end
