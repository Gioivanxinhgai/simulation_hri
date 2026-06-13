function plot_elbow()
    % ==========================================
    % THIẾT LẬP THẨM MỸ (Học từ figure1.m)
    % ==========================================
    fontsize = 15;
    linewidth_line = 1.5;
    
    % Định nghĩa mảng màu RGB chuẩn để khi in/xuất file màu không bị chói
    color_b = [0 0 1]; % Màu xanh dương
    color_r = [1 0 0]; % Màu đỏ
    
    set(0, 'DefaultAxesFontName', 'Times New Roman');
    set(0, 'DefaultAxesFontSize', fontsize); 
    set(0, 'DefaultTextFontName', 'Times New Roman');
    set(0, 'DefaultTextFontSize', fontsize);
    set(0, 'DefaultAxesLineWidth', 1.2); 
    
    % --- Đọc dữ liệu ---
    csv_file_path = 'elbow_data.csv';
    try
        df = readtable(csv_file_path);
    catch ME
        error('Không thể đọc tệp CSV: %s\n%s', csv_file_path, ME.message);
    end
    
    if ismember('K', df.Properties.VariableNames), M = df.K; else, M = df.M; end
    if ismember('Inertia', df.Properties.VariableNames), SSE = df.Inertia; else, SSE = df.Distortion; end
    
    % --- Thuật toán tìm điểm Elbow ---
    n_points = length(M);
    p1 = [M(1), SSE(1)]; 
    p2 = [M(end), SSE(end)];
    distances = zeros(n_points, 1);
    
    for i = 1:n_points
        p0 = [M(i), SSE(i)];
        num = abs((p2(2)-p1(2))*p0(1) - (p2(1)-p1(1))*p0(2) + p2(1)*p1(2) - p2(2)*p1(1));
        den = norm(p2 - p1);
        distances(i) = num / den;
    end
    [~, elbow_idx] = max(distances);
    
    % ==========================================
    % VẼ ĐỒ THỊ
    % ==========================================
    % Rộng 7.15 inches, Cao 3 inches
    fig = figure('Units', 'inches', 'Position', [1, 1, 7.15, 3], 'Color', 'w'); 
    
    ax1 = axes(fig);
    hold(ax1, 'on');
    
    % Vẽ đường line và marker (Dùng mã màu RGB và biến linewidth)
    plot(ax1, M, SSE, '-o', 'Color', color_b, 'LineWidth', linewidth_line, ...
         'MarkerSize', 5, 'MarkerFaceColor', color_b, 'MarkerEdgeColor', color_b);
         
    % 1. Vẽ điểm Elbow THẬT trên đồ thị (Viền đỏ, tâm xanh dương)
    plot(ax1, M(elbow_idx), SSE(elbow_idx), 'o', ...
         'MarkerSize', 10, ...
         'LineWidth', linewidth_line, ...
         'MarkerEdgeColor', color_r, ...    
         'MarkerFaceColor', color_b);       

    % 2. Tạo một điểm ẢO (tọa độ NaN) để thiết kế riêng cho Legend (Viền đỏ, tâm trắng)
    hLegendDummy = plot(ax1, NaN, NaN, 'o', ...
                        'MarkerSize', 10, ...
                        'LineWidth', linewidth_line, ...
                        'MarkerEdgeColor', color_r, ...    
                        'MarkerFaceColor', 'w', ...    
                        'DisplayName', 'Elbow point'); 

    % 3. Hiển thị Legend với font LaTeX
    legend(ax1, hLegendDummy, 'Location', 'northeast', ...
           'Interpreter', 'latex', ...
           'FontSize', 12, ... 
           'Box', 'on');
    
    % ==========================================
    % CÀI ĐẶT TRỤC & NHÃN THEO CHUẨN LATEX
    % ==========================================
    xlabel(ax1, 'Number of clusters', 'Interpreter', 'latex');
    ylabel(ax1, 'SSE', 'Interpreter', 'latex');
    
    % THAY ĐỔI TẠI ĐÂY:
    % 1. Đặt vị trí Tick tại các tọa độ thực tế để đồ thị chia khoảng cách đều đặn
    ax1.XTick = [121, 484, 968, 1452, 1936, 2420]; 
    
    % 2. Ghi đè nhãn hiển thị thành các giá trị làm tròn mà bạn mong muốn
    ax1.XTickLabel = {'121', '484', '969', '1454', '1939', '2424'};
    
    ax1.YTick = [0, 20, 40, 60, 80, 100]; 
    
    % Ép các số trên trục tọa độ dùng font toán học LaTeX
    set(ax1, 'TickLabelInterpreter', 'latex');
    
    ax1.XLim = [0, max(M)];
    y_margin = (max(SSE) - min(SSE)) * 0.05;
    ax1.YLim = [max(0, min(SSE) - y_margin), max(SSE) + y_margin]; 
    
    grid(ax1, 'off');
    box(ax1, 'on');
    
    ax1.Units = 'normalized';
    ax1.Position = [0.12, 0.22, 0.83, 0.62];  
   
    % ==========================================
    % XUẤT FILE EPS CHUẨN VECTOR
    % ==========================================
    drawnow;
    exportgraphics(fig, 'elbow_kmeans.eps', 'ContentType', 'vector', 'BackgroundColor', 'none');
    
    % Reset thiết lập mặc định
    set(0, 'DefaultAxesFontName', 'remove');
    set(0, 'DefaultAxesFontSize', 'remove');
    set(0, 'DefaultTextFontName', 'remove');
    set(0, 'DefaultTextFontSize', 'remove');
    set(0, 'DefaultAxesLineWidth', 'remove');
    
    fprintf('Đã vẽ và xuất xong file elbow_kmeans.eps.\n');
end