function plot_sampled_targets_only(sampled_csv_path)
    TARGETS_UNSCALED = [
         10,  10;   % Target 1 (10, 10)
        -13,   5;   % Target 2 (-13, 5)
          5, -13;   % Target 3 (5, -13)
    ];
    % Hằng số scale 
    SCALE_FACTOR = 0.03; 
    
    % Thiết lập font và kích thước cho đồ thị (Giữ nguyên)
    set(0, 'DefaultAxesFontName', 'Times New Roman');
    set(0, 'DefaultAxesFontSize', 12);
    set(0, 'DefaultTextFontName', 'Times New Roman');
    set(0, 'DefaultTextFontSize', 12);
    
    try        
        % Khởi tạo biểu đồ
        figure('Units','inches','Position',[0 0 4.3 4.3]); 
        ax = gca; 
        hold(ax, 'on');
        
        % 2. VẼ CÁC ĐIỂM TARGET ĐƯỢC LẤY MẪU BẰNG K-MEANS
        if exist(sampled_csv_path, 'file') == 2
            try
                % Đọc dữ liệu từ file CSV
                df_sampled = readtable(sampled_csv_path);
                
                if ismember('x', df_sampled.Properties.VariableNames) && ismember('y', df_sampled.Properties.VariableNames)
                    
                    % Áp dụng SCALE tương tự như Target gốc
                    x_sampled = df_sampled.x * SCALE_FACTOR; 
                    y_sampled = df_sampled.y * SCALE_FACTOR; 
                    
                    % Vẽ các điểm K-Means Target (dấu 'x' màu Đỏ đậm)
                    plot(ax, x_sampled, y_sampled, '.', 'MarkerEdgeColor', [0, 0.4470, 0.7410], ...
                         'MarkerSize', 4, 'LineWidth', 1.5, 'DisplayName', 'K-Means Sampled Targets');
                    disp(['Đã vẽ ', num2str(size(df_sampled, 1)), ' điểm K-Means Sampled Targets.']);
                         
                else
                    disp(['Cảnh báo: Tệp sampled targets ', sampled_csv_path, ' không chứa các cột ''x'' hoặc ''y''.']);
                end
            catch ME
                disp(['Lỗi khi xử lý tệp sampled targets ', sampled_csv_path, ': ', ME.message]);
            end
        else
            disp(['Cảnh báo: Không tìm thấy tệp sampled targets ', sampled_csv_path, '. Bỏ qua plotting.']);
        end
        % 3. Cấu hình và hiển thị biểu đồ
        xlabel(ax, 'x (m)'); 
        ylabel(ax, 'y (m)');         
        axis(ax, 'equal'); 
        axis(ax, 'tight');
        drawnow;
    catch ME
        % Xử lý lỗi khởi tạo (nếu có)
        disp(['Lỗi khi khởi tạo biểu đồ: ', ME.message]);
    end
end