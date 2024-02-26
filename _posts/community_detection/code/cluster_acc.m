function [out] = cluster_acc(true_labels, prediction, perm, varargin) 
    true_labels = true_labels(:);
    prediction = prediction(:);
    prediction = arrayfun(@(i) perm(i), prediction);
    out = mean(true_labels == prediction);
end

