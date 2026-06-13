% plot_trajectories_from_json_matlab.m
function plot_trajectories_from_json_matlab(json_file_path)
    % Tọa độ GỐC (CHƯA scale)
    TARGETS_UNSCALED = [
         10,  10;   % t=1: Góc trên phải (Sẽ vẽ là Target 2)
        -13,   5;   % t=2: Bên trái (Sẽ vẽ là Target 1)
          5, -13;   % t=3: Góc dưới phải (Sẽ vẽ là Target 3)
    ];
    SCALE_FACTOR = 0.03; 
    
    set(0, 'DefaultAxesFontName', 'Times New Roman');
    set(0, 'DefaultAxesFontSize', 12);
    set(0, 'DefaultTextFontName', 'Times New Roman');
    set(0, 'DefaultTextFontSize', 12);
    
    try
        json_str = fileread(json_file_path);
        trajectory_files = jsondecode(json_str);
        if ~iscell(trajectory_files), trajectory_files = cellstr(trajectory_files); end
        
        figure('Units','inches','Position',[0 0 4.3 4.3]); 
        ax = gca; 
        hold(ax, 'on');
        
        [json_dir, ~, ~] = fileparts(json_file_path);
        
        for i = 1:length(trajectory_files)
            file_name = trajectory_files{i};
            try
                csv_path = fullfile(json_dir, file_name);
                df = readtable(csv_path);
                
                if ismember('x', df.Properties.VariableNames) && ismember('y', df.Properties.VariableNames)
                    x_data = df.x * SCALE_FACTOR; 
                    y_data = df.y * SCALE_FACTOR; 
                    
                    color = get_trajectory_color_matlab(file_name);
                    linestyle = get_trajectory_linestyle_matlab(file_name);
                    
                    plot(ax, x_data, y_data, 'Color', color, 'LineStyle', linestyle, ...
                         'LineWidth', 1.0, 'DisplayName', file_name, 'Marker', 'none');
                    
                    % 📢 ĐÁNH DẤU ĐIỂM BẮT ĐẦU (Start Point)
                    if i == 1
                        % Vẽ đường kẻ chéo xuống bên trái
                        line_end_x = x_data(1) - 0.05;
                        line_end_y = y_data(1) - 0.05;
                        plot(ax, [x_data(1), line_end_x], [y_data(1), line_end_y], 'k-', 'LineWidth', 1.5);
                        
                        % Gắn chữ
                        text(ax, line_end_x - 0.01, line_end_y - 0.01, 'Start point', ...
                            'FontSize', 12, 'FontName', 'Times New Roman', ...
                            'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
                    end
                    % Vẽ chấm tròn đè lên đường kẻ
                    plot(ax, x_data(1), y_data(1), 'o', 'MarkerEdgeColor', 'k', ...
                        'MarkerFaceColor', 'w', 'MarkerSize', 8, 'LineWidth', 1.0);
                         
                else
                    disp(['Lỗi: Tệp ', csv_path, ' thiếu cột ''x'' hoặc ''y''.']);
                end
            catch ME
                disp(['Lỗi xử lý ', file_name, ': ', ME.message]);
            end
        end
        
        % 📢 THÊM 3 TARGET (CÓ ĐƯỜNG KẺ CHỈ DẪN Y HỆT HÌNH GỐC)
        for t = 1:size(TARGETS_UNSCALED, 1)
            target_x = TARGETS_UNSCALED(t, 1) * SCALE_FACTOR;
            target_y = TARGETS_UNSCALED(t, 2) * SCALE_FACTOR;
            
            % Ánh xạ tên và vị trí theo đúng hình gốc của bạn
            if t == 1
                % Điểm [10, 10] (Góc trên phải) -> Hình gốc gọi là "Target 2"
                label_text = 'Target 2';
                line_end_x = target_x - 0.06;
                line_end_y = target_y;
                text_x = line_end_x - 0.01;
                text_y = line_end_y;
                halign = 'right'; valign = 'middle';
            elseif t == 2
                % Điểm [-13, 5] (Bên trái) -> Hình gốc gọi là "Target 1"
                label_text = 'Target 1';
                line_end_x = target_x + 0.04;
                line_end_y = target_y + 0.04;
                text_x = line_end_x + 0.02;
                text_y = line_end_y - 0.02;
                halign = 'left'; valign = 'bottom';
            elseif t == 3
                % Điểm [5, -13] (Góc dưới phải) -> Hình gốc gọi là "Target 3"
                label_text = 'Target 3';
                line_end_x = target_x - 0.06;
                line_end_y = target_y;
                text_x = line_end_x - 0.01;
                text_y = line_end_y + 0.01;
                halign = 'right'; valign = 'middle';
            end
            
            % 1. Vẽ đường kẻ đen
            plot(ax, [target_x, line_end_x], [target_y, line_end_y], 'k-', 'LineWidth', 1.5);
            % 2. Vẽ dấu tròn đè lên đường kẻ
            plot(ax, target_x, target_y, 'o', 'MarkerEdgeColor', 'k', ...
                 'MarkerFaceColor', 'w', 'MarkerSize', 8, 'LineWidth', 1.0);
            % 3. Thêm chữ
            text(ax, text_x, text_y, label_text, ...
                'FontSize', 12, 'FontName', 'Times New Roman', ...
                'HorizontalAlignment', halign, 'VerticalAlignment', valign);
        end
        
        xlabel(ax, 'x (m)'); 
        ylabel(ax, 'y (m)'); 
        axis(ax, 'equal'); 
        
        % Tự động nới rộng khung hình thêm 18% để không bị cắt chữ và đường chỉ
        xl = xlim(ax); yl = ylim(ax);
        x_margin = (xl(2) - xl(1)) * 0.18;
        y_margin = (yl(2) - yl(1)) * 0.18;
        xlim(ax, [xl(1) - x_margin, xl(2) + x_margin]);
        ylim(ax, [yl(1) - y_margin, yl(2) + y_margin]);
        
        drawnow;
    catch ME
        disp(['Lỗi: ', ME.message]);
    end
end

function color = get_trajectory_color_matlab(file_name)
    tokens = regexp(file_name, 'trajectories_(\d+)\.csv', 'tokens', 'once');
    if ~isempty(tokens)
        traj_num = str2double(tokens{1});
        if traj_num >= 1 && traj_num <= 20, color = [0, 0.4470, 0.7410];
        elseif traj_num >= 61 && traj_num <= 80, color = [0, 0.4470, 0.7410];
        elseif traj_num >= 161 && traj_num <= 180, color = [0, 0.4470, 0.7410];
        else, color = [0, 0, 0]; end
    else, color = [0 0.5 0]; end
end

function linestyle = get_trajectory_linestyle_matlab(~), linestyle = '-'; end