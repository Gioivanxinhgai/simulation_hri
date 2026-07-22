import numpy as np
from config import C_V, F_C, K_H, K_P, M_ROB, M_ADM, K_ADM, D_ADM, N_SUB, DT

class PythonRobotDynamics:
    def __init__(self, dt=DT):
        self.dt = dt
        self.N_sub = N_SUB
        self.dt_sub = self.dt / self.N_sub
        
        # Admittance Control Parameters
        self.m_adm = M_ADM
        self.k_adm = K_ADM
        self.d_adm = D_ADM
        self.M_adm = np.diag([self.m_adm, self.m_adm, self.m_adm])
        self.K_adm = np.diag([self.k_adm, self.k_adm, self.k_adm])
        self.D_adm = np.diag([self.d_adm, self.d_adm, self.d_adm])
        self.M_adm_inv = np.linalg.inv(self.M_adm)
        
        # Robots Dynamics Parameters
        self.M_rob = M_ROB
        self.c_v = C_V
        self.F_c = F_C
        
        # Human stiffness parameter
        self.k_h = K_H
        
        # PD Control Parameters
        self.k_p = K_P
        self.LEADER_DECAY = 0.95  # Hệ số giảm năng lượng Admittance mỗi sub-step khi ở LEADER

        def _make_pd(k_p):
            k_d = 2.0 * np.sqrt(k_p * self.M_rob) - self.c_v   
            return np.diag([k_p, k_p, k_p]), np.diag([k_d, k_d, k_d])

        self.K_p_matrix, self.K_d_matrix = _make_pd(self.k_p)
        
        # States
        self.x_robot = np.zeros(3)
        self.dx_robot = np.zeros(3)
        self.x_adm = np.zeros(3)
        self.dx_adm = np.zeros(3)
        self.x_ref_prev = np.zeros(3)
        self.initialized = False

    def reset(self, x_init: np.ndarray = None):
        """Reset toàn bộ state về điểm xuất phát. Gọi explicit trước mỗi scenario mới."""
        if x_init is not None:
            self.x_robot = x_init.copy()
            self.x_ref_prev = x_init.copy()
        else:
            self.x_robot = np.zeros(3)
            self.x_ref_prev = np.zeros(3)
        self.dx_robot = np.zeros(3)
        self.x_adm   = np.zeros(3)
        self.dx_adm  = np.zeros(3)
        self.initialized = True
        print(f"[Dynamics] Reset -> x_init={self.x_robot}")

    def step(self, x_ref_in, x_h, is_leader=0.0, k_h=None):
        # Tính lực người f_h dựa trên intent x_h và vị trí robot hiện tại
        k_h_current = k_h if k_h is not None else self.k_h
        f_h = k_h_current * (x_h - self.x_robot)

        # Nếu chưa khởi tạo (lần đầu tiên), reset tĩnh về x_ref_in và return
        if not self.initialized:
            self.reset(x_ref_in)
            return self.x_robot.copy(), np.zeros(3), x_ref_in.copy(), f_h.copy()
            
        f_r = np.zeros(3)

        # Feedforward velocity tự đạo hàm của blended x_ref
        dx_ref_ff = (x_ref_in - self.x_ref_prev) / self.dt
        
        # PD Parameters
        K_p = self.K_p_matrix
        K_d = self.K_d_matrix

        # Sub-stepping
        for sub in range(1, self.N_sub + 1):
            alpha_interp = sub / self.N_sub
            x_ref_current = self.x_ref_prev + alpha_interp * (x_ref_in - self.x_ref_prev)
            
            # --- Admittance Control ---
            # Luôn tính toán biến đổi của Admittance dựa trên f_h
            damping_term = np.dot(self.D_adm, self.dx_adm)
            spring_term = np.dot(self.K_adm, self.x_adm)
            ddx_adm = np.dot(self.M_adm_inv, (f_h - damping_term - spring_term))
            
            self.dx_adm = self.dx_adm + ddx_adm * self.dt_sub
            self.x_adm = self.x_adm + self.dx_adm * self.dt_sub
            
            decay_factor = 1.0 - is_leader * (1.0 - self.LEADER_DECAY)
            self.x_adm = self.x_adm * decay_factor
            self.dx_adm = self.dx_adm * decay_factor
            
            x_d = x_ref_current + self.x_adm
            dx_d = dx_ref_ff + self.dx_adm
            
            # PD Tracking Control
            e_pos = x_d - self.x_robot
            e_vel =  0 - self.dx_robot
            f_r = np.dot(K_p, e_pos) + np.dot(K_d, e_vel)
            
            # Friction Model
            F_f = -np.tanh(100.0 * self.dx_robot) * (self.c_v * np.abs(self.dx_robot) + self.F_c)
            
            # Forward Dynamics
            # Ở LEADER, lực f_h không được coi là tác động vật lý chi phối robot -> blend giảm dần
            f_h_effective = (1.0 - is_leader) * f_h
            
            ddx_robot = (f_r + f_h_effective + F_f) / self.M_rob
            self.dx_robot = self.dx_robot + ddx_robot * self.dt_sub
            self.x_robot = self.x_robot + self.dx_robot * self.dt_sub
            
        self.x_ref_prev = x_ref_in.copy()
        
        return self.x_robot.copy(), f_r.copy(), x_d.copy(), f_h.copy()
