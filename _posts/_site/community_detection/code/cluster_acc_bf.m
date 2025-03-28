function [out] = cluster_acc_bf(true_labels, prediction, K) 
    potential_perm = perms(1:K);

    max_score = 0;
    for i=1:size(potential_perm, 1)
        perm = potential_perm(i, :);
        score = cluster_acc(true_labels, prediction, perm);
        if score> max_score
            max_score = score;
        end
    end
    out = max_score;
end

