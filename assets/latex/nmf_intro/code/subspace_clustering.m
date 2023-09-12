%% Clear all things
clc; clear; close all; path(pathdef);
N = 10;
d = 1;
D = 2;
U1 = rand(D, d);
U2 = rand(D, d);
S1 = 2*rand(d, N)-1;
S2 = 2*rand(d, N)-1;
noise1 = 1e-3*randn(D, N);
noise2 = 1e-3*randn(D, N);
X1 = U1*S1+noise1;
X2 = U2*S2+noise2;

figure()
colormap 'bone'
scatter(X1(1, :), X1(2, :), 'filled')
hold on
scatter(X2(1, :), X2(2, :), 'filled')

X = [X1, X2];
[U, S, V] = svds(X, 2);
figure();
imagesc(abs(V*V') <= 1e-3)
title('[binary] before permutation')
colormap('bone')
colorbar

figure();
imagesc(abs(V*V'))
title('[real] before permutation')
colorbar

perm_inds = randperm(2*N);
X1 = X(:, perm_inds);
[U, S, V] = svds(X1, 2);
figure();
imagesc(abs(V*V') <= 1e-5)
title('after permutation')
colormap('bone')
colorbar
