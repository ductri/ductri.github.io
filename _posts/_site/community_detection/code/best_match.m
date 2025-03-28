function [order] = best_match(true_labels, predicted_labels, k) 
    % Check the options structure.
    L = zeros(k, k);
    for i=1:k
        for j=1:k
            set1 = find(true_labels == i);
            set2 = find(predicted_labels == j);
            L(i, j) = -numel(intersect(set1, set2))/numel(set1);
        end
    end
    M = matchpairs(L, 100000000);
    ind = sub2ind([k, k], M(:, 1), M(:, 2));
    MSE = sum(L(ind));
    order = M(:, 1);
end

