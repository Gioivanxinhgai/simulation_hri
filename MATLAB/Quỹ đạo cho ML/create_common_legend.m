function create_common_legend()
    % Thiết lập thẩm mỹ
    set(0, 'DefaultTextInterpreter', 'latex');
    set(0, 'DefaultLegendInterpreter', 'latex');
    
    % Tạo một cửa sổ cực dẹt (Rộng 7 inches, cao chỉ 0.8 inches)
    % Nếu thấy 4 chú thích bị chật, bạn có thể tăng số 7 (ví dụ lên 8 hoặc 8.5)
    fig_leg = figure('Units', 'inches', 'Position', [1, 1, 8, 0.8], 'Color', 'w');
    ax = axes(fig_leg);
    hold(ax, 'on');
    
    % --- VẼ CÁC ĐƯỜNG ẢO (Dummy Plots) ĐỂ LẤY HANDLE CHO LEGEND ---
    % 1. Vùng 95% CI (Màu xám)
    ci_color = [0.85 0.85 0.85];
    p1 = fill(NaN, NaN, ci_color, 'EdgeColor', 'none');
    
    % 2. Đường Observation (Xanh dương, liền)
    p2 = plot(NaN, NaN, 'b-', 'LineWidth', 1.5);
    
    % 3. Đường Prediction (Đỏ, đứt nét)
    p3 = plot(NaN, NaN, 'r--', 'LineWidth', 1.5);
    
    % 4. [MỚI] Điểm đánh dấu thời gian t = 3.9 s (Chấm đen)
    p4 = plot(NaN, NaN, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
    
    % Tắt hoàn toàn khung trục đồ thị đi
    axis(ax, 'off');
    
    % Bóp nhỏ kích thước vật lý của trục tọa độ về gần bằng 0
    % Để exportgraphics không tính nó vào vùng viền bounding box
    ax.Position = [0.5 0.5 0.01 0.01];
    
    % --- TẠO LEGEND NẰM NGANG ---
    % [MỚI] Cập nhật mảng handle [p2, p3, p1, p4] và thêm tên '$t = 3.9$ s'
    lgd = legend([p2, p3, p1, p4], {'Ground Truth', 'Predicted', '95\% Confidence', '$t = 3.9$ s'}, ...
        'Orientation', 'horizontal', ... % Dàn hàng ngang
        'FontSize', 16, ...
        'Box', 'on', ...
        'Location', 'none'); % Đổi thành 'none' để ngắt Legend ra khỏi trục
    
    % Tăng độ dày viền (outline) của hộp Legend và ép màu viền đen tuyền
    lgd.LineWidth = 1.5;  % Viền dày bằng nét đồ thị
    lgd.EdgeColor = 'k';  % Viền đen tuyền (Black)
    
    % Ép MATLAB cập nhật giao diện để tính toán chính xác kích thước (width/height) của Legend
    drawnow; 
    
    % Tự động căn giữa Legend vào chính giữa Figure
    lgd.Units = 'normalized';
    lgd.Position(1) = 0.5 - lgd.Position(3)/2; % Căn giữa theo trục ngang
    lgd.Position(2) = 0.5 - lgd.Position(4)/2; % Căn giữa theo trục dọc
    
    % Xuất ra file .eps
    drawnow;
    exportgraphics(fig_leg, 'Common_Legend.eps', 'ContentType', 'vector', 'BackgroundColor', 'none');
    
    fprintf('Đã tạo thành công file Common_Legend.eps với 4 thành phần chú thích!\n');
end