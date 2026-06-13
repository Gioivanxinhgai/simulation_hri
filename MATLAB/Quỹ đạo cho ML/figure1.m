% load('S2.mat');
t = 0:1e-3:60;
t = t';
%% Color
color1 = [0 0 1]; 
color2 = [0,0.9,0.9];
color3 = [1 0 0];
color4 = [0.00,1.00,0.00];
color5 = [0.93,0.69,0.13];
color6 = [0.00, 0.00, 1.00];
color_r = [0, 0, 0];
color_b = [0.1 0.8 0.3];
%% Define
fontsize = 20;
linewidth = 1.5;
%%
b1Hx = b1H(1:2:length(t)*2-1);
b1Hy = b1H(2:2:length(t)*2);

b1Lx = b1L(1:2:length(t)*2-1);
b1Ly = b1L(2:2:length(t)*2);

b2Hx = b2H(1:2:length(t)*2-1);
b2Hy = b2H(2:2:length(t)*2);

b2Lx = b2L(1:2:length(t)*2-1);
b2Ly = b2L(2:2:length(t)*2);

dqx = dq(:,1);
dqy = dq(:,2);

dtaux = dtau(1:2:length(t)*2-1);
dtauy = dtau(2:2:length(t)*2);

dtau1x = dtau1(1:2:length(t)*2-1);
dtau1y = dtau1(2:2:length(t)*2);

dx2dx = dx2d(1:2:length(t)*2-1);
dx2dy = dx2d(2:2:length(t)*2);

dx2d1x = dx2d1(1:2:length(t)*2-1);
dx2d1y = dx2d1(2:2:length(t)*2);

e1x = e1(:,1);
e1y = e1(:,2);

e2x = e2(:,1);
e2y = e2(:,2);

qx = q(:,1);
qy = q(:,2);

qrx = qr(1:2:length(t)*2-1);
qry = qr(2:2:length(t)*2);

taux = tau(1:2:length(t)*2-1);
tauy = tau(2:2:length(t)*2);

tau1x = tau1(1:2:length(t)*2-1);
tau1y = tau1(2:2:length(t)*2);

% x1 = x(:,1);
% x2 = x(:,2);

x2dx = x2d(1:2:length(t)*2-1);
x2dy = x2d(2:2:length(t)*2);

x2d1x = x2d1(1:2:length(t)*2-1);
x2d1y = x2d1(2:2:length(t)*2);

% x1r = xr(1:2:length(t)*2-1);
% x2r = xr(2:2:length(t)*2);

Xi1mx = Xi1m(:,1);
Xi1my = Xi1m(:,2);

Xi2mx = Xi2m(:,1);
Xi2my = Xi2m(:,2);

Xi1x = Xi1(:,1);
Xi1y = Xi1(:,2);

Xi2x = Xi2(:,1);
Xi2y = Xi2(:,2);

%% e1
figure;
hold on;
% ===== Phase time =====
t1 = 20;
t2 = 40;
t3 = t(end);

yl = [-0.5 0.5];   % chỉnh theo ylim mong muốn

% ===== Background phases =====
p1 = patch([0 t1 t1 0], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 0.9 1], 'EdgeColor','none','FaceAlpha',0.5);

p2 = patch([t1 t2 t2 t1], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 1 0.9], 'EdgeColor','none','FaceAlpha',0.5);

p3 = patch([t2 t3 t3 t2], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 0.9 1], 'EdgeColor','none','FaceAlpha',0.5);
l1 = plot(t,b1Hx,'Color', color_r, 'LineStyle','-.','LineWidth',linewidth);
l2 = plot(t,e1x,'Color', color1, 'LineStyle','-','LineWidth',linewidth);
l3 = plot(t,e1y,'Color', color3, 'LineStyle','-','LineWidth',linewidth);
l4 = plot(t,b1Lx,'Color', color_r, 'LineStyle','-.','LineWidth',linewidth);

ylabel('$\mathbf{e}_1$ (rad)', 'Interpreter', 'latex');
xlabel('Time (s)', 'Interpreter', 'latex');
set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman');
legend([l1 l2 l3 p1 p2], ...
       {'Bound','$e_{11}$','$e_{12}$', ...
        'UC','C'}, ...
       'Interpreter','latex', ...
       'FontSize',fontsize, ...
       'Location','best');
ylim([-0.5,0.5]);

%% e2
figure;
hold on;
% ===== Phase time =====
t1 = 20;
t2 = 40;
t3 = t(end);

yl = [-2 2];   % chỉnh theo ylim mong muốn

% ===== Background phases =====
p1 = patch([0 t1 t1 0], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 0.9 1], 'EdgeColor','none','FaceAlpha',0.5);

p2 = patch([t1 t2 t2 t1], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 1 0.9], 'EdgeColor','none','FaceAlpha',0.5);

p3 = patch([t2 t3 t3 t2], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 0.9 1], 'EdgeColor','none','FaceAlpha',0.5);
l1 = plot(t,b2Hx,'Color', color_r, 'LineStyle','-.','LineWidth',linewidth);
l2 = plot(t,e2x,'Color', color1, 'LineStyle','-','LineWidth',linewidth);
l3 = plot(t,e2y,'Color', color3, 'LineStyle','-','LineWidth',linewidth);
l4 = plot(t,b2Lx,'Color', color_r, 'LineStyle','-.','LineWidth',linewidth);

ylabel('$\mathbf{e}_2$ (rad/s)', 'Interpreter', 'latex');
xlabel('Time (s)', 'Interpreter', 'latex');
set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman');
legend([l1 l2 l3 p1 p2], ...
       {'Upper bound','${e}_{21}$','${e}_{22}$', ...
        'UC','C'}, ...
       'Interpreter','latex', ...
       'FontSize',fontsize, ...
       'Location','best');
ylim([-2,2]);
%% q
figure;
hold on;
% ===== Phase time =====
t1 = 20;
t2 = 40;
t3 = t(end);

yl = [-3.5 3.5];   % chỉnh theo ylim mong muốn

% ===== Background phases =====
p1 = patch([0 t1 t1 0], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 0.9 1], 'EdgeColor','none','FaceAlpha',0.5);

p2 = patch([t1 t2 t2 t1], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 1 0.9], 'EdgeColor','none','FaceAlpha',0.5);

p3 = patch([t2 t3 t3 t2], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 0.9 1], 'EdgeColor','none','FaceAlpha',0.5);
l1 = plot(t,B1H,'Color', color_r, 'LineStyle','-.','LineWidth',linewidth);
l2 = plot(t,qx,'Color', color1, 'LineStyle','-','LineWidth',linewidth);
l3 = plot(t,qy,'Color', color3, 'LineStyle','-','LineWidth',linewidth);
l4 = plot(t,qrx,'Color', color2, 'LineStyle','--','LineWidth',linewidth);
l5 = plot(t,qry,'Color', color4, 'LineStyle','--','LineWidth',linewidth);
l6 = plot(t,B1L,'Color', color_r, 'LineStyle','-.','LineWidth',linewidth);

ylabel('$\mathbf{q}$ (rad)', 'Interpreter', 'latex');
xlabel('Time (s)', 'Interpreter', 'latex');
set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman');
legend([l1 l2 l3 l4 l5 p1 p2], ...
       {'Upper bound','$x_{11}$','$x_{12}$', '$y_{d1}$','$y_{d2}$', ...
        'UC','C'}, ...
       'Interpreter','latex', ...
       'FontSize',fontsize, ...
       'Location','best');
ylim([-3.5,3.5]);

%% dq
figure;
hold on;
% ===== Phase time =====
t1 = 20;
t2 = 40;
t3 = t(end);

yl = [-3.5 3.5];   % chỉnh theo ylim mong muốn

% ===== Background phases =====
p1 = patch([0 t1 t1 0], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 0.9 1], 'EdgeColor','none','FaceAlpha',0.5);

p2 = patch([t1 t2 t2 t1], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 1 0.9], 'EdgeColor','none','FaceAlpha',0.5);

p3 = patch([t2 t3 t3 t2], [yl(1) yl(1) yl(2) yl(2)], ...
      [0.9 0.9 1], 'EdgeColor','none','FaceAlpha',0.5);
l1 = plot(t,B2H,'Color', color_r, 'LineStyle','-.','LineWidth',linewidth);
l2 = plot(t,dqx,'Color', color1, 'LineStyle','-','LineWidth',linewidth);
l3 = plot(t,dqy,'Color', color3, 'LineStyle','-','LineWidth',linewidth);
l4 = plot(t,x2dx,'Color', color2, 'LineStyle','--','LineWidth',linewidth);
l5 = plot(t,x2dy,'Color', color4, 'LineStyle','--','LineWidth',linewidth);
l6 = plot(t,B2L,'Color', color_r, 'LineStyle','-.','LineWidth',linewidth);

ylabel('$\mathbf{\dot{q}}$ (rad/s)', 'Interpreter', 'latex');
xlabel('Time (s)', 'Interpreter', 'latex');
set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman');
legend([l1 l2 l3 l4 l5 p1 p2], ...
       {'Upper bound','$x_{21}$','$x_{22}$', '$u_{11}$','$u_{12}$', ...
        'UC','C'}, ...
       'Interpreter','latex', ...
       'FontSize',fontsize, ...
       'Location','best');
ylim([-3.5,3.5]);
%% x2d
figure;
hold on;

l1 = plot(t,u1M,'Color', color_r, 'LineStyle','--','LineWidth',linewidth);
l2 = plot(t,x2dx,'Color', color1, 'LineStyle','-','LineWidth',linewidth);
l3 = plot(t,x2dy,'Color', color3, 'LineStyle','-','LineWidth',linewidth);
l4 = plot(t,x2d1x,'Color', color2, 'LineStyle','--','LineWidth',linewidth);
l5 = plot(t,x2d1y,'Color', color4, 'LineStyle','--','LineWidth',linewidth);
l6 = plot(t,u1m,'Color', color_r, 'LineStyle','--','LineWidth',linewidth);

ylabel('$\mathbf{u}_1$ (rad/s)', 'Interpreter', 'latex');
xlabel('Time (s)', 'Interpreter', 'latex');
legend([l1 l2 l3 l4 l5 l6], ...
       {'Upper bound','$u_{11}$','$u_{12}$', '$u^*_{11}$','$u^*_{12}$', 'Lower bound'}, ...
       'Interpreter','latex', ...
       'FontSize',fontsize, ...
       'Location','best');
ylim([-4,4]);
%% dx2d
figure;
hold on;

l1 = plot(t,u1R,'Color', color_r, 'LineStyle','--','LineWidth',linewidth);
l2 = plot(t,dx2dx,'Color', color1, 'LineStyle','-','LineWidth',linewidth);
l3 = plot(t,dx2dy,'Color', color3, 'LineStyle','-','LineWidth',linewidth);
l4 = plot(t,dx2d1x,'Color', color2, 'LineStyle','--','LineWidth',linewidth);
l5 = plot(t,dx2d1y,'Color', color4, 'LineStyle','--','LineWidth',linewidth);
l6 = plot(t,u1r,'Color', color_r, 'LineStyle','--','LineWidth',linewidth);

ylabel('$\mathbf{\dot{u}}_1$ (rad/s$^2$)', 'Interpreter', 'latex');
xlabel('Time (s)', 'Interpreter', 'latex');
set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman');
legend([l1 l2 l3 l4 l5 l6], ...
       {'Upper bound','$\dot{u}_{11}$','$\dot{u}_{12}$', '$\dot{u}^*_{11}$','$\dot{u}^*_{12}$', 'Lower bound'}, ...
       'Interpreter','latex', ...
       'FontSize',fontsize, ...
       'Location','best');
ylim([-120,120]);
%% tau
figure;
hold on;
l1 = plot(t,u2M,'Color', color_r, 'LineStyle','--','LineWidth',linewidth);
l2 = plot(t,taux,'Color', color1, 'LineStyle','-','LineWidth',linewidth);
l3 = plot(t,tauy,'Color', color3, 'LineStyle','-','LineWidth',linewidth);
l4 = plot(t,tau1x,'Color', color2, 'LineStyle','--','LineWidth',linewidth);
l5 = plot(t,tau1y,'Color', color4, 'LineStyle','--','LineWidth',linewidth);
l6 = plot(t,u2m,'Color', color_r, 'LineStyle','--','LineWidth',linewidth);

ylabel('$\mathbf{u}_2$ (Nm)', 'Interpreter', 'latex');
xlabel('Time (s)', 'Interpreter', 'latex');
set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman');
legend([l1 l2 l3 l4 l5 l6], ...
       {'Upper bound','$u_{21}$','$u_{22}$', '$u^*_{21}$','$u^*_{22}$', 'Lower bound'}, ...
       'Interpreter','latex', ...
       'FontSize',fontsize, ...
       'Location','best');
ylim([-20,20]);
%% dtau
figure;
hold on;

l1 = plot(t,u2R,'Color', color_r, 'LineStyle','--','LineWidth',linewidth);
l2 = plot(t,dtaux,'Color', color1, 'LineStyle','-','LineWidth',linewidth);
l3 = plot(t,dtauy,'Color', color3, 'LineStyle','-','LineWidth',linewidth);
l4 = plot(t,dtau1x,'Color', color2, 'LineStyle','--','LineWidth',linewidth);
l5 = plot(t,dtau1y,'Color', color4, 'LineStyle','--','LineWidth',linewidth);
l6 = plot(t,u2r,'Color', color_r, 'LineStyle','--','LineWidth',linewidth);

ylabel('$\mathbf{\dot{u}}_2$ (Nm/s)', 'Interpreter', 'latex');
xlabel('Time (s)', 'Interpreter', 'latex');
set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman');
legend([l1 l2 l3 l4 l5 l6], ...
       {'Upper bound','$\dot{u}_{11}$','$\dot{u}_{12}$', '$\dot{u}^*_{11}$','$\dot{u}^*_{12}$', 'Lower bound'}, ...
       'Interpreter','latex', ...
       'FontSize',fontsize, ...
       'Location','best');
ylim([-250,250]);
%% Xi1
figure;
hold on;
l1 = plot(t,Xi1mx,'Color', color1, 'LineStyle','-','LineWidth',linewidth);
l2 = plot(t,Xi1my,'Color', color3, 'LineStyle','-','LineWidth',linewidth);
l3 = plot(t,Xi1x,'Color', color2, 'LineStyle','--','LineWidth',linewidth);
l4 = plot(t,Xi1y,'Color', color4, 'LineStyle','--','LineWidth',linewidth);

ylabel('$\mathbf{\Xi}_1$', 'Interpreter', 'latex');
xlabel('Time (s)', 'Interpreter', 'latex');
set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman');
legend([l1 l2 l3 l4], ...
       {'$\hat{\Xi}_{11}$','$\hat{\Xi}_{12}$', '$\Xi_{11}$','$\Xi_{12}$'}, ...
       'Interpreter','latex', ...
       'FontSize',fontsize, ...
       'Location','best');
ylim([-2,2]);
%% Xi2
figure;
hold on;

l1 = plot(t,Xi2mx,'Color', color1, 'LineStyle','-','LineWidth',linewidth);
l2 = plot(t,Xi2my,'Color', color3, 'LineStyle','-','LineWidth',linewidth);
l3 = plot(t,Xi2x,'Color', color2, 'LineStyle','--','LineWidth',linewidth);
l4 = plot(t,Xi2y,'Color', color4, 'LineStyle','--','LineWidth',linewidth);

ylabel('$\mathbf{\Xi}_2$', 'Interpreter', 'latex');
xlabel('Time (s)', 'Interpreter', 'latex');
set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman');
legend([l1 l2 l3 l4], ...
       {'$\hat{\Xi}_{21}$','$\hat{\Xi}_{22}$', '$\Xi_{21}$','$\Xi_{22}$'}, ...
       'Interpreter','latex', ...
       'FontSize',fontsize, ...
       'Location','best');
ylim([-10,10]);
% %% xy
% figure;
% hold on;
% plot(x1,x2,'Color', color1, 'LineStyle','-','LineWidth',linewidth);
% plot(x1r,x2r,'Color', color3, 'LineStyle','--','LineWidth',linewidth);
% 
% ylabel('$x$ (m)', 'Interpreter', 'latex');
% xlabel('$y$ (m)', 'Interpreter', 'latex');
% set(gca, 'FontSize', fontsize, 'FontName', 'Times New Roman');
% % xlim([0 300]);
% % ylim([-12,8]);