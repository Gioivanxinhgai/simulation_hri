% 1. Thiết lập biến và cấu hình chung
CSV_PATH = fullfile("training_log.csv");

set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 16);
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultTextFontSize', 16);
set(0, 'DefaultLegendFontSize', 16);
set(0, 'DefaultLineLineWidth', 1.0); 

% 2. Gọi hàm thực thi
plot_train(CSV_PATH);

% 3. Định nghĩa hàm
function plot_train(csv_path)

    if ~isfile(csv_path)
        error('FileNotFoundError: Không tìm thấy file: %s', csv_path);
    end

    try
        df = readtable(csv_path);
    catch ME
        error('Lỗi khi đọc file CSV: %s', ME.message);
    end

    required_cols = {'epoch', 'train_loss', 'val_mse'};
    if ~all(ismember(required_cols, df.Properties.VariableNames))
        error('ValueError: File CSV thiếu cột: epoch, train_loss, val_mse');
    end

    % --- PHẦN QUAN TRỌNG: TRÍCH XUẤT DỮ LIỆU ---
    epochs = df.epoch;
    train_loss = df.train_loss; % <-- Dòng này khắc phục lỗi "Unrecognized variable"
    val_mse = df.val_mse;

    % ─── Hình 1: Train Loss theo Epoch ───────────────────────────────
    fig1 = figure('Units','inches','Position',[0 0 4.5 2.5], 'Name', 'Train Loss');
    ax1 = axes(fig1);
    
    % Vẽ Train Loss (Màu xanh lá [0 0.6 0] như bạn muốn)
    plot(ax1, epochs, train_loss, '.', 'Color', [0 0.6 0], 'LineWidth', 1.0, ...
         'DisplayName', 'Negative Log-Likelihood'); 
    
    xlabel(ax1, 'epoch', 'FontSize', 16, 'FontName', 'Times New Roman');
    ylabel(ax1, 'Training Loss', 'FontSize', 16, 'FontName', 'Times New Roman');
    legend(ax1, 'show', 'Location', 'northeast');
    
    axis(ax1, 'tight');
    
    x_range1 = diff(ax1.XLim); 
    y_range1 = diff(ax1.YLim);
    
    ax1.XLim = ax1.XLim + [-0.005 * x_range1, 0.005 * x_range1];
    ax1.YLim = ax1.YLim + [0.05 * y_range1, 0.05 * y_range1];

    % ─── Hình 2: Validation MSE theo Epoch ───────────────────────────────
    fig2 = figure('Units','inches','Position',[0 0 4.5 2.5], 'Name', 'Val MSE');
    ax2 = axes(fig2);
    
    % Vẽ Val MSE (Cũng để màu xanh lá cho đồng bộ)
    plot(ax2, epochs, val_mse, '.', 'Color',  [0 0.6 0], 'LineWidth', 1.0, ...
         'DisplayName', 'MSE'); 

    xlabel(ax2, 'epochs', 'FontSize', 16, 'FontName', 'Times New Roman');
    ylabel(ax2, 'MSE Loss', 'FontSize', 16, 'FontName', 'Times New Roman');
    legend(ax2, 'show', 'Location', 'northeast');
    
    axis(ax2, 'tight');
    
    x_range2 = diff(ax2.XLim);
    y_range2 = diff(ax2.YLim);
    
    ax2.XLim = ax2.XLim + [-0.005 * x_range2, 0.005 * x_range2];
    ax2.YLim = ax2.YLim + [0.05 * y_range2, 0.05 * y_range2];
 
end