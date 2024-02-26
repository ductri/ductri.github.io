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
M = dirichlet_rnd(0.3*ones(1, K), N);
% M(:, 1:K) = eye(K);
% M(:, K+1:2*K) = eye(K);
[~, node_labels] = max(M);
B = diag([0.9, 0.9, 1,]);
P = M'*B*M;
seed = rand(size(P));
A = tril(seed <= P, -1) * 1.0;
G = graph(A, 'lower');

radius = 10; 
centroid_x = [1, 300, 300];
centroid_y = [1, 1, 300];

Xdata = radius*rand(size(node_labels)) + arrayfun(@(l) centroid_x(l), node_labels);
Ydata = radius*rand(size(node_labels)) + arrayfun(@(l) centroid_y(l), node_labels);
base_colors = [1 0 0; 0 1 0; 0 0 1];
node_colors = M'*base_colors;

% set(groot, 'defaultAxesTickLabelInterpreter','latex');
% % set(groot, 'defaultLegendInterpreter','latex');
% set(groot, 'DefaultLineMarkerSize', 9)
% set(groot, 'DefaultLineLineWidth', 1.4)
% set(groot, 'DefaultAxesFontSize', 14);
figure()
p = plot(G);
% p.NodeColor = node_colors;
print('../images/network-demo-x.png', '-dpng')
