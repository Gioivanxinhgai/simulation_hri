function plot_prediction_comparison_triple(csv_file_path)
    %% 1. Định nghĩa thông số (Đồng bộ outline và nét vẽ)
    SCALE_FACTOR = 0.03; 
    fontsize = 20;
    linewidth = 1.5;
    axes_linewidth = 1.2; % Outline dày dặn như bạn mong muốn
    
    color_obs  = [0 0 1];          
    color_pred = [1 0 0];          
    color_fill = [0.825 0.825 0.825]; 
    
    %% 2. Đọc và Scale Dữ liệu
    try
        df = readtable(csv_file_path);
    catch ME
        error('Không thể đọc tệp CSV: %s\n%s', csv_file_path, ME.message);
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
    
    %% 3. THIẾT LẬP LAYOUT (TiledLayout)
    % Kích thước 15x5 inches để đảm bảo khi gộp lại các hình vẫn đủ độ lớn
    fig = figure('Units','inches','Position',[0 0 15 5], 'Color', 'w'); 
    tlo = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    %% 4. Tile 1: Trục X(t)
    nexttile;
    hold on;
    p1 = fill([t; flipud(t)], [lower_x; flipud(upper_x)], color_fill, ...
         'EdgeColor', 'none', 'FaceAlpha', 0.9); % Shaded region nét (Alpha 0.9)
    l1 = plot(t, obs_x, 'Color', color_obs, 'LineStyle', '-', 'LineWidth', linewidth); 
    l2 = plot(t, pred_x, 'Color', color_pred, 'LineStyle', '--', 'LineWidth', linewidth);  
    
    % Nhãn 2 dòng: Dòng 1 là đơn vị, Dòng 2 là (a) in đậm
    xlabel({'Time (s)', '\textbf{(a)}'}, 'Interpreter', 'latex');
    ylabel('$x$ (m)', 'Interpreter', 'latex');
    
    set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman', 'LineWidth', axes_linewidth);
    axis tight; box on; grid off;
    set(gca, 'XTick', [0, 2.5, 5, 7.5, 10]);
           
    %% 5. Tile 2: Trục Y(t)
    nexttile;
    hold on;
    fill([t; flipud(t)], [lower_y; flipud(upper_y)], color_fill, ...
         'EdgeColor', 'none', 'FaceAlpha', 0.9);
    plot(t, obs_y, 'Color', color_obs, 'LineStyle', '-', 'LineWidth', linewidth);
    plot(t, pred_y, 'Color', color_pred, 'LineStyle', '--', 'LineWidth', linewidth);  
    
    xlabel({'Time (s)', '\textbf{(b)}'}, 'Interpreter', 'latex');
    ylabel('$y$ (m)', 'Interpreter', 'latex');
    
    set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman', 'LineWidth', axes_linewidth);
    axis tight; box on; grid off;
    set(gca, 'XTick', [0, 2.5, 5, 7.5, 10]);
           
    %% 6. Tile 3: Quỹ đạo 2D Y(x)
    nexttile;
    hold on;
    l_obs_2d = plot(obs_x, obs_y, 'Color', color_obs, 'LineStyle', '-', 'LineWidth', linewidth);
    l_pred_2d = plot(pred_x, pred_y, 'Color', color_pred, 'LineStyle', '--', 'LineWidth', linewidth);  
    
    xlabel({'$x$ (m)', '\textbf{(c)}'}, 'Interpreter', 'latex');
    ylabel('$y$ (m)', 'Interpreter', 'latex');
    
    set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman', 'LineWidth', axes_linewidth);
    axis tight; box on; grid off;
    
    % Nới trục y lên 0.15 theo yêu cầu cũ
    y_min = min(min(obs_y), min(pred_y));
    ylim([y_min, 0.15]);
    set(gca, 'YTick', [0, 0.05, 0.1, 0.15]);

    %% 7. LEGEND TỔNG QUÁT (Layout Level)
    % Sử dụng các handle l1, l2 và p1 để đại diện cho toàn bộ 3 hình
    lgd = legend([l1, l2, p1], {'Observation', 'Prediction', '95\% Confidence Interval'}, ...
           'Interpreter', 'latex', ...
           'FontSize', fontsize, ...
           'Orientation', 'horizontal');
    
    % Đặt legend lên phía trên cùng của toàn bộ layout
    lgd.Layout.Tile = 'north'; 
    lgd.Box = 'on'; % Có khung bao quanh legend để nhìn rõ ràng hơn
end