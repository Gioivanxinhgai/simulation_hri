function matlab_server()
    PORT = 25000;
    DT = 12 / 189;
    MAX_STEPS = 100000;
    % Admittance Control Parameters
    M_adm = diag([1.0, 1.0, 1.0]);
    k_adm_val = 150;
    K_adm = diag([k_adm_val, k_adm_val, k_adm_val]);
    d_adm_val = 25
    D_adm = diag([d_adm_val, d_adm_val, d_adm_val]);
    % Robots Dynamics Parameters
    M_rob = 0.8;
    c_v = 0.5;
    F_c = 2.0;
    % PD Control Parmaeters
    k_p_val = 2000;
    K_p = diag([k_p_val, k_p_val, k_p_val]);
    k_d_val = 2*0.7*sqrt(k_p_val*M_rob);
    K_d = diag([k_d_val, k_d_val, k_d_val]);

    LEADER_DECAY = 0.95;
    % Robot initial states
    x_robot = zeros(3, 1);
    dx_robot = zeros(3, 1);
    x_adm = zeros(3, 1);
    dx_adm = zeros(3, 1);
    x_ref_prev = zeros(3, 1);
    initialized = false;
    
    fprintf('SHARED CONTROL SIMULATION\n');
    server_socket = tcpserver("0.0.0.0", PORT, "Timeout", 60, "ByteOrder", "little-endian");
    % Wait for connection
    while server_socket.NumBytesAvailable == 0
        pause(0.05);
        if server_socket.Connected
            break;
        end
    end
    fprintf('[Server] Python connected.\n\n');

    step = 0;
    log_x = zeros(MAX_STEPS, 3);
    log_xref = zeros(MAX_STEPS, 3);
    log_fr = zeros(MAX_STEPS, 3);
    log_fh = zeros(MAX_STEPS, 3);

    while step < MAX_STEPS
        bytes_needed = 56;
        wait_count = 0;
        while server_socket.NumBytesAvailable < bytes_needed
            pause(0.001);
            wait_count = wait_count + 1;
            if wait_count > 120000 %Timeout
                fprintf('[Server] Timeout waiting for data. Stopping.\n');
                return;
            end
        end
        % Read Data
        raw = read(server_socket, 7, "double");
        if isempty(raw) || any(isnan(raw(1:3)))
            break;
        end
        x_ref_in = raw(1:3)';
        f_h = raw(4:6)';
        is_leader = raw(7);
        % Reset scenario
        if ~initialized || norm(x_ref_in - x_ref_prev) > 0.1
            x_robot = x_ref_in;
            x_adm = zeros(3, 1);
            dx_adm = zeros(3, 1);
            dx_robot = zeros(3, 1);
            x_ref_prev = x_ref_in;
            initialized = true;
        end
        step = step + 1;
        % Sub-stepping
        N_sub = 100;
        dt_sub = DT / N_sub;

        for sub = 1:N_sub
            alpha_interp = sub / N_sub;
            x_ref_current = x_ref_prev + alpha_interp * (x_ref_in - x_ref_prev);

            % LUÔN tính toán Admittance để đảm bảo Robot compliant với lực người (Human-centered)
            ddx_adm = M_adm \ (f_h - D_adm * dx_adm - K_adm * x_adm);
            dx_adm = dx_adm + ddx_adm * dt_sub;
            x_adm = x_adm + dx_adm * dt_sub;
            x_d = x_ref_current + x_adm;
            dx_d = dx_adm;

            % PD Tracking Control
            e_pos = x_d - x_robot;
            e_vel = dx_d - dx_robot;
            f_r = K_p * e_pos + K_d * e_vel;

            % Friction Model
            F_f = -tanh(100*dx_robot) .* (c_v * abs(dx_robot) + F_c);

            % Forward Dynamics
            ddx_robot = (f_r + F_f + f_h) / M_rob;
            dx_robot = dx_robot + ddx_robot * dt_sub;
            x_robot = x_robot + dx_robot * dt_sub;
        end

        x_ref_prev = x_ref_in;

        % --- SEND RESPONSE ---
        response = [x_robot(:); f_r(:)];
        write(server_socket, response, "double");

        % --- LOGGING ---
        log_x(step, :) = x_robot';
        log_xref(step, :) = x_ref_in';
        log_fr(step, :) = f_r';
        log_fh(step, :) = f_h';
    end

    fprintf('\n[Server] Simulation done. Total steps: %d\n', step);
end