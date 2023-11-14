%% Clear all things
clc; clear; close all; path(pathdef);
addpath('~/code/matlab/common/prob_tools')

n = 10000;
N = sample_normal(n);
N = rand(n, 1);
X = sample_normal(n);
% X = rand(n, 1);
Y = X + N; %% X = Y/2 - N/2;

figure()
scatter(X, Y);
xlabel('x')
ylabel('y')

figure()
scatter(Y-(X'*Y/(X'*X))*X, X);
xlabel('y-f(x)')
ylabel('x')

figure()
scatter(Y, -0.5*Y+X);
xlabel('y')
ylabel('x-f(y)')

