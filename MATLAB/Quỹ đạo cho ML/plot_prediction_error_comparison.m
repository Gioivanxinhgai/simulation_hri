function plot_prediction_error_comparison(csv_file_A, csv_file_B)
    % --- Thiết lập kiểu dáng (Style) Chung ---
    set(0, 'DefaultAxesFontName', 'Times New Roman');
    set(0, 'DefaultAxesFontSize', 18);
    set(0, 'DefaultTextFontName', 'Times New Roman');
    set(0, 'DefaultTextFontSize', 18);
    set(0, 'DefaultLegendFontSize', 14); 
    set(0, 'DefaultAxesLineWidth', 1.2); 
    
    % --- Đọc dữ liệu cho 2 Mô hình ---
    [tA, oxA, pxA, ~, ~, oyA, pyA, ~, ~] = read_and_scale(csv_file_A);
    [tB, oxB, pxB, ~, ~, oyB, pyB, ~, ~] = read_and_scale(csv_file_B);
    
    % --- Tính toán Sai số tuyệt đối (Absolute Error) ---
    % Error = |Prediction - Observation|
    err_xA = abs(pxA - oxA);
    err_yA = abs(pyA - oyA);
    
    err_xB = abs(pxB - oxB);
    err_yB = abs(pyB - oyB);
    
    % 💡 THÊM: Tìm sai số lớn nhất trong toàn bộ dữ liệu để làm mốc chung cho trục tung
    max_err = max([max(err_xA), max(err_xB), max(err_yA), max(err_yB)]);
    % Tạo khoảng giới hạn từ 0 đến Max + 5% khoảng trống phía trên cho đẹp mắt
    y_limits = [0, max_err * 1.05]; 

    % ==========================================
    % 1. SO SÁNH SAI SỐ TRÊN TRỤC X(t)
    % ==========================================
    fig_errX = figure('Units','inches','Position',[0 3 6 4.5], 'Color', 'w'); 
    axX = axes(fig_errX); hold(axX, 'on');
    
    % Vẽ sai số của Model A (màu xanh, nét liền) và Model B (màu đỏ, nét đứt)
    plot(axX, tA, err_xA, 'b-', 'LineWidth', 1.5, 'DisplayName', 'GPR+K-Means'); 
    plot(axX, tB, err_xB, 'r-', 'LineWidth', 1.5, 'DisplayName', 'GPR');  
    
    xlabel(axX, 'Time (s)'); 
    ylabel(axX, 'Error (m)');
    
    % Hiển thị Legend nằm ngang ở giữa viền trên
    legend(axX, 'show', 'Location', 'northwest', 'Orientation', 'horizontal');  
    axis(axX, 'tight'); 
    
    % 💡 THÊM: Ép tỷ lệ trục tung chung
    ylim(axX, y_limits);
    
    box(axX, 'on');
    
    % ==========================================
    % 2. SO SÁNH SAI SỐ TRÊN TRỤC Y(t)
    % ==========================================
    fig_errY = figure('Units','inches','Position',[6.5 3 6 4.5], 'Color', 'w'); 
    axY = axes(fig_errY); hold(axY, 'on');
    
    % Vẽ sai số của Model A (màu xanh, nét liền) và Model B (màu đỏ, nét đứt)
    plot(axY, tA, err_yA, 'b-', 'LineWidth', 1.5, 'DisplayName', 'GPR+K-Means'); 
    plot(axY, tB, err_yB, 'r-', 'LineWidth', 1.5, 'DisplayName', 'GPR');  
    
    xlabel(axY, 'Time (s)'); 
    ylabel(axY, 'Error (m)');
    
    legend(axY, 'show', 'Location', 'northwest', 'Orientation', 'horizontal');  
    axis(axY, 'tight'); 
    
    % 💡 THÊM: Ép tỷ lệ trục tung chung
    ylim(axY, y_limits);
    
    box(axY, 'on');
    
    % 💡 THÊM: Liên kết 2 trục tung để đảm bảo tỷ lệ luôn được đồng bộ nếu bạn dùng chuột thu phóng đồ thị
    linkaxes([axX, axY], 'y');

    % Xóa cấu hình Default sau khi chạy xong
    set(0, 'DefaultAxesLineWidth', 'remove');
end

% =======================================================
% HÀM HỖ TRỢ: Đọc file CSV và Scale dữ liệu
% =======================================================
function [t, ox, px, lx, ux, oy, py, ly, uy] = read_and_scale(filepath)
    try 
        df = readtable(filepath); 
    catch ME
        error('Lỗi đọc CSV: %s', ME.message); 
    end
    
    % Scale factor lấy từ code gốc của bạn
    SCALE_FACTOR = 0.03;
    
    t = df.t;
    ox = df.orig_x * SCALE_FACTOR;  
    px = df.pred_x * SCALE_FACTOR;
    lx = df.lower_x_95ci * SCALE_FACTOR; 
    ux = df.upper_x_95ci * SCALE_FACTOR;
    
    oy = df.orig_y * SCALE_FACTOR;  
    py = df.pred_y * SCALE_FACTOR;
    ly = df.lower_y_95ci * SCALE_FACTOR; 
    uy = df.upper_y_95ci * SCALE_FACTOR;
end