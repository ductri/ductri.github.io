%% Clear all things
clc; clear; close all; path(pathdef);

for i=1:1111
    theta = 10000;
    n = 100;
    Xs = randn(n,1) + theta;

    theta_hat = mean(Xs);

    err(i) = (theta - theta_hat)^2;
end
mean(err)
