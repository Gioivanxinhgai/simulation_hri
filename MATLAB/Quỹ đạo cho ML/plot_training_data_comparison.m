function plot_training_data_comparison(csv_full, csv_sampled)
    % ==========================================
    % THIẾT LẬP THẨM MỸ TỔNG THỂ (Chuẩn Vector / LaTeX)
    % ==========================================
    fontsize = 18; 
    
    set(0, 'DefaultAxesFontName', 'Times New Roman');
    set(0, 'DefaultAxesFontSize', fontsize);
    set(0, 'DefaultTextFontName', 'Times New Roman');
    set(0, 'DefaultTextFontSize', fontsize); 
    set(0, 'DefaultAxesLineWidth', 1.2); 
    
    % [ĐÃ SỬA] Cập nhật Scale Factor mới
    SCALE_FACTOR = 0.1; 
    
    % Màu sắc RGB
    color_b = [0, 0.4470, 0.7410];
    
    configs = {
        struct('file', csv_full, 'name', 'Full Points', 'color', color_b);
        struct('file', csv_sampled, 'name', 'Sampled Points', 'color', color_b)
    };
    
    % --- VẼ TỪNG TẬP DỮ LIỆU ---
    for i = 1:length(configs)
        cfg = configs{i};
        fig = figure('Name', cfg.name, 'Units','inches','Position',[1 + (i-1)*5, 1, 5.1, 4.65], 'Color', 'w'); 
        
        ax = axes(fig);
        hold(ax, 'on');
        
        set(ax, 'FontName', 'Times New Roman', 'FontSize', fontsize);
        
        if exist(cfg.file, 'file') == 2
            try
                df = readtable(cfg.file);
                
                if ismember('x', df.Properties.VariableNames) && ismember('y', df.Properties.VariableNames)
                    x_scaled = df.x * SCALE_FACTOR; 
                    y_scaled = df.y * SCALE_FACTOR; 
                    
                    scatter(ax, x_scaled, y_scaled, 5, cfg.color, 'filled');
                    disp(['Đã vẽ ', num2str(size(df, 1)), ' điểm cho ', cfg.name]);
                else
                    disp(['Cảnh báo: Tệp ', cfg.file, ' không chứa các cột ''x'' hoặc ''y''.']);
                end
            catch ME
                disp(['Lỗi khi xử lý tệp ', cfg.file, ': ', ME.message]);
            end
        else
            disp(['Cảnh báo: Không tìm thấy tệp ', cfg.file, '. Bỏ qua plotting.']);
        end
        
        % --- BỔ SUNG CÁC ĐIỂM VÀ NHÃN (CHỈ DÀNH CHO HÌNH FULL) ---
        if i == 1
            % [ĐÃ SỬA] Nhân tọa độ các Target với tỉ lệ (10/3)
            start_pt = [0, 0];
            target1  = [-1.3, 0.5]; 
            target2  = [1.0, 1.0];   
            target3  = [0.5, -1.3]; 
           
            plot(ax, start_pt(1), start_pt(2), 'ko', 'MarkerFaceColor', 'w', 'MarkerSize', 8, 'LineWidth', 1.5);
            plot(ax, target1(1), target1(2), 'ko', 'MarkerFaceColor', 'w', 'MarkerSize', 8, 'LineWidth', 1.5);
            plot(ax, target2(1), target2(2), 'ko', 'MarkerFaceColor', 'w', 'MarkerSize', 8, 'LineWidth', 1.5);
            plot(ax, target3(1), target3(2), 'ko', 'MarkerFaceColor', 'w', 'MarkerSize', 8, 'LineWidth', 1.5);
            
            % [ĐÃ SỬA] Scale các khoảng cách (offset) của nhãn chữ tương ứng
            text_fs = 16;
            text(ax, start_pt(1)-0.53, start_pt(2)-0.07, 'Initial', 'FontSize', text_fs, 'Interpreter', 'latex');
            text(ax, start_pt(1)-0.60, start_pt(2)-0.20, 'position', 'FontSize', text_fs, 'Interpreter', 'latex');
            text(ax, target1(1)+0.07, target1(2)+0.10, 'Target 1', 'FontSize', text_fs, 'Interpreter', 'latex');
            text(ax, target2(1)-0.67, target2(2)+0.07, 'Target 2', 'FontSize', text_fs, 'Interpreter', 'latex');
            text(ax, target3(1)+0.04, target3(2)+0.09, 'Target 3', 'FontSize', text_fs, 'Interpreter', 'latex');
        end
        
        % ==========================================
        % SỬ DỤNG LATEX CHO NHÃN TRỤC (X, Y) VÀ CÁC SỐ TRÊN TRỤC
        % ==========================================
        xlabel(ax, '$x$ (m)', 'Interpreter', 'latex'); 
        ylabel(ax, '$y$ (m)', 'Interpreter', 'latex'); 
        
        axis(ax, 'equal'); 
        
        % [ĐÃ SỬA] Thiết lập giới hạn trục cố định mới (cũ là -0.45 đến 0.35)
        xlim(ax, [-1.5, 1.2]); 
        ylim(ax, [-1.5, 1.2]);
        
        box(ax, 'off');
        
        % [ĐÃ SỬA] FORMAT TRỤC X & Y VỚI CÁC BƯỚC NHẢY MỚI (Step = 0.5)
        ticks_arr = [-1.5, -1.0, -0.5, 0, 0.5, 1.0];
        tick_labels = {'-1.5', '-1.0', '-0.5', '0', '0.5', '1.0'};
        
        set(ax, 'XTick', ticks_arr);
        set(ax, 'XTickLabel', tick_labels);
        
        set(ax, 'YTick', ticks_arr);
        set(ax, 'YTickLabel', tick_labels);
        
        set(ax, 'TickLabelInterpreter', 'latex');
        
        % ==========================================
        % TỐI ƯU MARGIN 
        % ==========================================
        ax.Units = 'normalized';
        ax.Position = [0.15, 0.15, 0.80, 0.80];  
        drawnow;
        
        % ==========================================
        % XUẤT ẢNH .EPS ĐỊNH DẠNG VECTOR
        % ==========================================
        safe_name = strrep(cfg.name, ' ', '_'); 
        exportgraphics(fig, [safe_name, '.eps'], 'ContentType', 'vector', 'BackgroundColor', 'none');
        
    end
    
    set(0, 'DefaultAxesFontName', 'remove');
    set(0, 'DefaultAxesFontSize', 'remove');
    set(0, 'DefaultTextFontName', 'remove');
    set(0, 'DefaultTextFontSize', 'remove');
    set(0, 'DefaultAxesLineWidth', 'remove');
    
    fprintf('Đã xuất thành công các file .eps chuẩn Vector.\n');
end