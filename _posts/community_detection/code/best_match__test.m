%% Clear all things
clc; clear; close all; path(pathdef);

labels = [3, 2, 1, 3, 3, 3, 3, 3, 1, 1, 1];
predict = [1, 2, 3, 1, 1, 1, 1, 1, 3, 3, 3];
% predict = [3, 2, 1, 3, 3, 3, 3, 3, 1, 1, 1];

order = best_match(labels, predict, 3);

