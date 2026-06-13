function ls()
    % ---------------------------------------------------------
    % 1. THIẾT LẬP STYLE
    % ---------------------------------------------------------
    fontsize = 16; % Điều chỉnh cỡ chữ cho phù hợp với khung gộp
    linewidth = 1.5;
    markersize = 7;
    
    set(0, 'DefaultAxesFontName', 'Times New Roman');
    set(0, 'DefaultTextFontName', 'Times New Roman');
    set(0, 'DefaultAxesFontSize', fontsize); 
    set(0, 'DefaultLegendFontSize', 15);
    
    % ---------------------------------------------------------
    % 2. DỮ LIỆU
    % ---------------------------------------------------------
    x_full = [7.8351, 7.6775, 7.5919, 7.5314, 7.4829, 7.4390, 7.3863, 7.2970, 7.0789, 4.2440];
    y_full = [8.0235, 7.8197, 7.7253, 7.6675, 7.6273, 7.5917, 7.5286, 7.2430, 7.2351, 4.0627];
    x_kmeans = [9.9065, 9.6635, 9.5380, 9.4571, 9.3958, 9.3175, 9.2440, 9.1655, 8.9007, 5.2320];
    y_kmeans = [10.3170, 9.9831, 9.8273, 9.7280, 9.6394, 9.5625, 9.4527, 8.7431, 9.0089, 4.8154];
    t_index = 1:10; 
    
    color_full = [0, 0.4470, 0.7410];   % Xanh dương
    color_kmeans = [1.0, 0.5, 0.0];     % Cam 
    
    % ---------------------------------------------------------
    % 3. KHỞI TẠO CỬA SỔ CHUNG VÀ BỐ CỤC (LAYOUT)
    % ---------------------------------------------------------
    % Tạo 1 figure rộng hơn (ví dụ 8 x 4 inches)
    fig = figure('Units', 'inches', 'Color', 'w', 'Position', [1, 2, 8, 4]);
    
    % Sử dụng tiledlayout (1 hàng, 2 cột) để quản lý Legend và Margin dễ hơn subplot
    tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % ---------------------------------------------------------
    % HÌNH TRÁI (a): LENGTHSCALE X
    % ---------------------------------------------------------
    ax1 = nexttile(tl); 
    hold(ax1, 'on'); box(ax1, 'on'); grid(ax1, 'on');
    p1 = plot(ax1, t_index, x_full, '-o', 'Color', color_full, 'LineWidth', linewidth, ...
        'MarkerSize', markersize, 'MarkerFaceColor', color_full, 'MarkerEdgeColor', color_full);
    p2 = plot(ax1, t_index, x_kmeans, '-o', 'Color', color_kmeans, 'LineWidth', linewidth, ...
        'MarkerSize', markersize, 'MarkerFaceColor', color_kmeans, 'MarkerEdgeColor', color_kmeans);
    
    ylabel(ax1, 'Length Scale', 'Interpreter', 'latex'); 
    % Dùng cell array {} để ngắt dòng xlabel, thêm chữ (a) ở dưới
    xlabel(ax1, {'Input Feature Index', '(a)'}, 'Interpreter', 'latex'); 
    
    % [MỚI] Khóa giới hạn trục để ôm sát điểm dữ liệu và hiển thị đủ các số trên trục tung
    xlim(ax1, [0 10]);
    ylim(ax1, [2 12.1]); 
    
    % [MỚI] Chỉ định chính xác các giá trị hiển thị trên trục
    set(ax1, 'XTick', [0, 2, 4, 6, 8, 10]);
    set(ax1, 'YTick', [2, 4, 6, 8, 10, 12]);
    set(ax1, 'TickLabelInterpreter', 'latex');
    
    % ---------------------------------------------------------
    % HÌNH PHẢI (b): LENGTHSCALE Y
    % ---------------------------------------------------------
    ax2 = nexttile(tl);
    hold(ax2, 'on'); box(ax2, 'on'); grid(ax2, 'on');
    p3 = plot(ax2, t_index, y_full, '-o', 'Color', color_full, 'LineWidth', linewidth, ...
        'MarkerSize', markersize, 'MarkerFaceColor', color_full, 'MarkerEdgeColor', color_full);
    p4 = plot(ax2, t_index, y_kmeans, '-o', 'Color', color_kmeans, 'LineWidth', linewidth, ...
        'MarkerSize', markersize, 'MarkerFaceColor', color_kmeans, 'MarkerEdgeColor', color_kmeans);
    
    ylabel(ax2, 'Length Scale', 'Interpreter', 'latex');
    xlabel(ax2, {'Input Feature Index', '(b)'}, 'Interpreter', 'latex');
    
    % [MỚI] Khóa giới hạn trục để ôm sát điểm dữ liệu
    xlim(ax2, [0 10]);
    ylim(ax2, [2 12.1]); 
    
    % [MỚI] Chỉ định chính xác các giá trị hiển thị trên trục
    set(ax2, 'XTick', [0, 2, 4, 6, 8, 10]);
    set(ax2, 'YTick', [2, 4, 6, 8, 10, 12]);
    set(ax2, 'TickLabelInterpreter', 'latex');
    
    % ---------------------------------------------------------
    % 4. THIẾT LẬP LEGEND CHUNG (NẰM TRÊN CÙNG)
    % ---------------------------------------------------------
    % Tạo legend từ trục 1 nhưng đẩy nó ra quản lý chung bởi tiledlayout
    lgd = legend(ax1, [p2, p1], {'KMGPR', 'GPR'}, ...
        'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'on');
    
    % Ép vị trí của Legend lên phía Bắc (Trên cùng) của toàn bộ Figure
    lgd.Layout.Tile = 'North'; 
    
    % ---------------------------------------------------------
    % 5. XUẤT FILE EPS
    % ---------------------------------------------------------
    drawnow;
    exportgraphics(fig, 'Lengthscale_Combined.eps', 'ContentType', 'vector', 'BackgroundColor', 'none');
    
    % Dọn dẹp Workspace
    set(0, 'DefaultAxesFontName', 'remove');
    set(0, 'DefaultTextFontName', 'remove');
    set(0, 'DefaultAxesFontSize', 'remove');
    set(0, 'DefaultLegendFontSize', 'remove');
    
    fprintf('Đã xuất thành công file Lengthscale_Combined.eps!\n');
end