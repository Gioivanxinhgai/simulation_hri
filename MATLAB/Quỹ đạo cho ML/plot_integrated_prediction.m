function plot_integrated_prediction(csv_file_path)
    SCALE_FACTOR = 0.03; 
    
    % --- Thiết lập kiểu dáng (Style) Chung ---
    set(0, 'DefaultAxesFontName', 'Times New Roman');
    set(0, 'DefaultAxesFontSize', 18);
    set(0, 'DefaultTextFontName', 'Times New Roman');
    set(0, 'DefaultTextFontSize', 18);
    set(0, 'DefaultLegendFontSize', 15); 
    set(0, 'DefaultAxesLineWidth', 1.2); 
    
    % --- Đọc dữ liệu ---
    try
        df = readtable(csv_file_path);
    catch ME
        error('Không thể đọc tệp CSV: %s\n%s', csv_file_path, ME.message);
    end
    
    % --- Trích xuất và Scale dữ liệu ---
    t = df.t;
    obs_x = df.orig_x * SCALE_FACTOR;
    pred_x = df.pred_x * SCALE_FACTOR;
    lower_x = df.lower_x_95ci * SCALE_FACTOR;
    upper_x = df.upper_x_95ci * SCALE_FACTOR;
    
    obs_y = df.orig_y * SCALE_FACTOR;
    pred_y = df.pred_y * SCALE_FACTOR;
    lower_y = df.lower_y_95ci * SCALE_FACTOR;
    upper_y = df.upper_y_95ci * SCALE_FACTOR;
    
    % =======================================================
    % VẼ TÍCH HỢP TRÊN CÙNG 1 ĐỒ THỊ
    % =======================================================
    fig = figure('Units','inches','Position',[0 0 8 5], 'Color', 'w'); 
    ax = axes(fig);
    hold(ax, 'on');
    
    % Định nghĩa màu sắc
    color_x      = 'k';                               
    color_x_pred = [1.0, 0.5, 0.0];                   
    color_y      = 'b';                               
    color_y_pred = [0.9, 0.3, 0.1];                   
    
    % 1. Vẽ vùng 95% CI 
    % Lớp x: Ẩn khỏi legend
    fill(ax, [t; flipud(t)], [lower_x; flipud(upper_x)], [0.85 0.85 0.85], ...
         'EdgeColor', 'none', 'FaceAlpha', 0.9, 'HandleVisibility', 'off');
         
    % Lớp y: Hiển thị trên legend. 
    % 💡 LƯU Ý: Phải dùng '\%' vì đang bật Interpreter LaTeX
    fill(ax, [t; flipud(t)], [lower_y; flipud(upper_y)], [0.85 0.85 0.85], ...
         'EdgeColor', 'none', 'FaceAlpha', 0.9, 'DisplayName', '95\% CI');
         
    % 2. Vẽ 4 đường dữ liệu chính
    plot(ax, t, obs_x, '-', 'Color', color_x, 'LineWidth', 1.5, 'DisplayName', '$x$'); 
    plot(ax, t, pred_x, '--', 'Color', color_x_pred, 'LineWidth', 1.5, 'DisplayName', '$\hat{x}$');  
    
    plot(ax, t, obs_y, '-', 'Color', color_y, 'LineWidth', 1.5, 'DisplayName', '$y$');
    plot(ax, t, pred_y, '--', 'Color', color_y_pred, 'LineWidth', 1.5, 'DisplayName', '$\hat{y}$');  
    % 3. Cấu hình trục, nhãn và ĐƯỜNG LƯỚI
    xlabel(ax, 'Time (s)');
    ylabel(ax, 'Position (m)');
    grid(ax, 'off');
    
    xlim(ax, [min(t), max(t)]);
    
    % =======================================================
    % ÔM SÁT ĐỒ THỊ VÀ CẤU HÌNH TRỤC TUNG (Y-AXIS)
    % =======================================================
    min_val = min([min(lower_x), min(lower_y)]);
    max_val = max([max(upper_x), max(upper_y)]);
    
    margin = (max_val - min_val) * 0.01;
    y_lower_limit = min_val - margin;
    y_upper_limit = max_val + margin;
    
    ylim(ax, [y_lower_limit, y_upper_limit]);
    
    % Tính toán 4 mốc chia trên trục tung
    y_tick_values = linspace(y_lower_limit, y_upper_limit, 4);
    yticks(ax, y_tick_values);
    
    % 💡 Xử lý lỗi -0.0
    y_labels = arrayfun(@(v) sprintf('%.1f', v), y_tick_values, 'UniformOutput', false);
    y_labels = strrep(y_labels, '-0.0', '0.0');
    yticklabels(ax, y_labels);
    
    box(ax, 'on');
    
    % =======================================================
    % CĂN CHỈNH LEGEND NỬA TRONG NỬA NGOÀI
    % =======================================================
    % MATLAB sẽ tự động ghép các phần tử có tên (DisplayName) vào Legend
    lgd = legend(ax, 'show', 'Orientation', 'horizontal', 'Interpreter', 'latex', 'Box', 'on');
    
    drawnow; 
    
    set(ax, 'Units', 'normalized');
    set(lgd, 'Units', 'normalized');
    
    ax_pos = get(ax, 'Position');  
    lgd_pos = get(lgd, 'Position'); 
    
    new_x = ax_pos(1) + (ax_pos(3) - lgd_pos(3)) / 2;
    new_y = ax_pos(2) + ax_pos(4) - (lgd_pos(4) / 2);
    
    set(lgd, 'Position', [new_x, new_y, lgd_pos(3), lgd_pos(4)]);
    
    set(0, 'DefaultAxesLineWidth', 'remove');
end