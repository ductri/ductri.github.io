%% Clear all things
clc; clear; close all; path(pathdef);
addpath('~/code/matlab/common/prob_tools/')
addpath('~/code/matlab/common/PGD')
addpath('~/code/matlab/common/prox_ops')
addpath('~/code/matlab/common/')
addpath('~/code/matlab/sdmmv_clean')
addpath('~/code/matlab/sdmmv_clean/fw_core')

N = 100;
K = 3;
M = dirichlet_rnd(0.2*ones(1, K), N);
M(:, 1:K) = eye(K);
M(:, K+1:2*K) = eye(K);
[~, node_labels] = max(M);
B = diag([0.9, 0.8, 0.9]);
P = M'*B*M;
seed = rand(size(P));
A = tril(seed <= P, -1) * 1.0;
G = graph(A, 'lower');

radius = 80; 
centroid_x = [1, 50, 100];
centroid_y = [1, 100, 1];

Xdata = radius*rand(size(node_labels)) + arrayfun(@(l) centroid_x(l), node_labels);
Ydata = radius*rand(size(node_labels)) + arrayfun(@(l) centroid_y(l), node_labels);
base_colors = [1 0 0; 0 1 0; 0 0 1];
node_colors = M'*base_colors;

figure();
% p = plot(G, 'Xdata', Xdata, 'Ydata', Ydata);
p = plot(G);
% p.NodeColor = node_colors;
% title('raw network')
% colorbar
print('../images/network-demo-gt-nocolor.png', '-dpng')

figure();
% p = plot(G, 'Xdata', Xdata, 'Ydata', Ydata);
p = plot(G);
p.NodeColor = node_colors;
title('true membership')
colorbar
print('../images/network-demo-gt.png', '-dpng')



% idx = spectralcluster(A, K);
% fprintf('----- Spectral Clustering on raw A ----- \n')
% perm = best_match(node_labels, idx, K);
% fprintf('Acc: %f \n', cluster_acc(node_labels, idx, perm));
% fprintf('Acc (brute force): %f \n', cluster_acc_bf(node_labels, idx, K));
%
% fprintf('----- Spectral Clustering on singular vectors -----\n')
A = tril(A)+tril(A,-1)';
[U, ~] = eigs(A, K);
% idx = spectralcluster(U, K);
% perm = best_match(node_labels, idx, K);
% fprintf('Acc: %f \n', cluster_acc(node_labels, idx, perm));

% fprintf('Acc (brute force): %f \n', cluster_acc_bf(node_labels, idx, K));

fprintf(' ---------- Our SDMMV ------- ')
[C_hat, Tracking_fw] = fw(U', 'lambda', 0, 'backend', 'matlab', 'debug', true);
% figure()
% plot(Tracking_fw.InnerTracking.obj);
% title('Convex Self-dictionary objective')
% print('./../images/sdmmv-obj.png', '-dpng')

[v, lambdaHat] = maxk(vecnorm(C_hat, Inf, 2), K, 1);
W_hat = U(lambdaHat, :)';

ops = struct;
ops.debug = true;
ops.max_iters = 500;
ops.f_fn = @(H) norm(U' - W_hat * H, 'fro')^2;
p_fn = @(x, h) proj_simplex_matrix(x);
step_size = 0.1;
init_point = rand(K, N);
init_point = init_point ./ sum(init_point);
g_fn = @(H) W_hat'*(W_hat*H - U');
[H_hat, Tracking] = pgd(g_fn, p_fn, step_size, init_point, ops);
% figure()
% plot(Tracking.obj)
% title('Constrained LS')
% fprintf('Constrained LS obj value: %f \n', Tracking.obj(end));

[~, my_pred] = max(H_hat);
perm = best_match(node_labels, my_pred, K);
fprintf('Acc: %f \n', cluster_acc(node_labels, my_pred, perm));

fprintf('Acc (brute force): %f \n', cluster_acc_bf(node_labels, my_pred, K));

figure();
% [~, pred_node_labels] = max(H_hat);
% pred_Xdata = radius*rand(size(node_labels)) + arrayfun(@(l) centroid_x(l), pred_node_labels);
% pred_Ydata = radius*rand(size(node_labels)) + arrayfun(@(l) centroid_y(l), pred_node_labels);
% p = plot(G, 'Xdata', pred_Xdata, 'Ydata', pred_Ydata);
p = plot(G, 'layout', 'Force');
pred_node_colors = H_hat'*base_colors;
p.NodeColor = pred_node_colors;
title('pred membership')
colorbar
print('../images/network-demo-pred.png', '-dpng')

fprintf('%f\n', mse_measure_norm(M', H_hat'))
