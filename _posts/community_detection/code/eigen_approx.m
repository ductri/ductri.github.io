%% Clear all things
clc; clear; close all; path(pathdef);
addpath('~/code/matlab/common/prob_tools/')
addpath('~/code/matlab/common/PGD')
addpath('~/code/matlab/common/prox_ops')
addpath('~/code/matlab/common/')
addpath('~/code/matlab/sdmmv_clean')
addpath('~/code/matlab/sdmmv_clean/fw_core')

errors = [];
list_N = [100:200:2000];

for j=1:numel(list_N)
    j
    for i=1:20
        N = list_N(j);
        K = 3;
        M = dirichlet_rnd(0.3*ones(1, K), N);
        [~, node_labels] = max(M);
        B = diag([0.9, 0.9, 1]);
        P = M'*B*M;
        seed = rand(size(P));
        A = tril(seed <= P, -1) * 1.0;
        A = tril(A)+tril(A,-1)';

        [V, ~] = eigs(P, K);
        [V_hat, ~] = eigs(A, K);
        % errors(j, i) = mse_measure_norm(V, V_hat);
        errors(j, i) = subspace(V, V_hat);
    end
end

set(groot, 'defaultAxesTickLabelInterpreter','latex');
% set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'DefaultLineMarkerSize', 9)
set(groot, 'DefaultLineLineWidth', 1.4)
set(groot, 'DefaultAxesFontSize', 14);
figure()
plot(list_N, mean(errors, 2))
xlabel('#nodes')
ylabel('angle')
print('../images/eigen.png', '-dpng')


