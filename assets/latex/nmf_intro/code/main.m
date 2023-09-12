%% Clear all things
clc; clear; close all; path(pathdef);

I = 1000;
J = 1000;
F = 62;
W = zeros(I, F);
H = zeros(F, J);

% The first component
W(:, 1) = zeros(I, 1);
W(round(I/3):round(2*I/3), 1) = 1;
% W(:, 1) = W(:, 1) / sum(W(:, 1));
H(1, :) = zeros(1, J);
H(1, round(J/2)-2:round(J/2)+2) = 1;

X1 = W(:, 1)*H(1, :);
figure()
image(X1, 'CDataMapping','scaled')
colormap 'bone'
colorbar
title('first component')

% The second component
W(:, 2) = zeros(I, 1);
W(round(I/2)-2:round(I/2)+2, 2) = 1;
H(2, round(2*J/5):round(4*J/5)) = 1;

X2 = W(:, 2)*H(2, :);
figure()
image(X2, 'CDataMapping','scaled')
colormap 'bone'
colorbar
title('second component')

% The `third` component
A = double(im2gray(imresize(imread('sky.png'), [I, J])));
% [U, S, V] = svds(A, 50);
[U, V] = nnmf(A, F-2);
% A_low = U*V;
scaling_factor = sum(U, 1);
W(:, 3:end) = U./scaling_factor;
H(3:end, :) = scaling_factor'.*V;

X3 = W(:, 3:end)*H(3:end, :);
figure();
image(X3, 'CDataMapping','scaled');
colormap 'bone'
colorbar
title('the third component')

X12 = W(:, 1)*H(1, :) + W(:, 2)*H(2, :);
colormap 'bone'
figure();
image(X12, 'CDataMapping','scaled')
colorbar
title('the 1+2 component')

alpha = 1e-2;
X = (1-alpha)*W(:, 1:2)*H(1:2, :) + alpha*W(:, 3:end)*H(3:end, :);
figure()
image(X, 'CDataMapping','scaled')
colormap 'bone'
colorbar
title('the whole signal')


alpha = 1e-2;
X = (1-alpha)*W(:, 1:2)*H(1:2, :) + alpha*A;
figure()
image(X, 'CDataMapping','scaled')
colormap 'bone'
colorbar
title('the whole signal 2')

[U, V] = nnmf(X, 2);

figure()
image(U*V, 'CDataMapping','scaled')
colormap 'bone'
colorbar
title('the extracted 2 components')
