function plot_prediction_comparison(csv_file_path)
    SCALE_FACTOR = 0.03; 
    
    % --- Thiết lập kiểu dáng (Style) ---
    set(0, 'DefaultAxesFontName', 'Times New Roman');
    set(0, 'DefaultAxesFontSize', 20);
    set(0, 'DefaultTextFontName', 'Times New Roman');
    set(0, 'DefaultTextFontSize', 20);
    
    % --- Đọc dữ liệu ---
    try
        df = readtable(csv_file_path);
    catch ME
        error('Không thể đọc tệp CSV: %s', ME.message);
    end
    
    t = df.t;
    obs_x = df.orig_x * SCALE_FACTOR;
    pred_x = df.pred_x * SCALE_FACTOR;
    lower_x = df.lower_x_95ci * SCALE_FACTOR;
    upper_x = df.upper_x_95ci * SCALE_FACTOR;
    obs_y = df.orig_y * SCALE_FACTOR;
    pred_y = df.pred_y * SCALE_FACTOR;
    lower_y = df.lower_y_95ci * SCALE_FACTOR;
    upper_y = df.upper_y_95ci * SCALE_FACTOR;
    
    % --- Vẽ Hình ---
    fig = figure('Units','inches','Position',[0 0 8 6], 'Color', 'w'); 
    ax = axes(fig);
    hold(ax, 'on');
    
    % 1. Vẽ vùng 95% CI 
    % 💡 Thay đổi: Chỉ cho phép 1 vùng hiển thị trong Legend (HandleVisibility)
    f1 = fill(ax, [t; flipud(t)], [lower_x; flipud(upper_x)], [0.825 0.825 0.825], ...
         'EdgeColor', 'none', 'FaceAlpha', 1.0, 'DisplayName', '95\% CI');
    
    fill(ax, [t; flipud(t)], [lower_y; flipud(upper_y)], [0.825 0.825 0.825], ...
         'EdgeColor', 'none', 'FaceAlpha', 1.0, 'HandleVisibility', 'off');
    
    % 2. Vẽ các đường chính
    p2 = plot(ax, t, obs_x,  '-',  'Color', 'k',       'LineWidth', 1.2, 'DisplayName', '$x_r$');
    p1 = plot(ax, t, pred_x, '--', 'Color', [1 0.5 0], 'LineWidth', 1.5, 'DisplayName', '$\hat{x}$'); 
    p4 = plot(ax, t, obs_y,  '-',  'Color', 'b',       'LineWidth', 1.2, 'DisplayName', '$y_r$');
    p3 = plot(ax, t, pred_y, '--', 'Color', [0.825 0.2 0], 'LineWidth', 1.5, 'DisplayName', '$\hat{y}$');
    
    % --- Thiết lập trục ---
    xlabel(ax, 'Time (s)');
    ylabel(ax, 'Position (m)');
    
    axis(ax, 'tight');
    box(ax, 'on');
    set(ax, 'LineWidth', 1.2); 
    
    % --- Legend nằm ngang phía trên ---
    % 💡 Cập nhật: Thêm f1 vào danh sách hiển thị
    lgd = legend(ax, [p1, p2, p3, p4, f1], 'Orientation', 'horizontal', ...
                 'Location', 'northoutside', 'Interpreter', 'latex', ...
                 'FontSize', 18); % Giảm nhẹ size legend để đủ chỗ cho 5 item
    
    % Căn chỉnh lại vị trí để legend không bị khuất
    ax.Position = [0.12 0.12 0.8253 0.72]; 
end