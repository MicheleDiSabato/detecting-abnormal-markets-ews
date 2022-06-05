function [bestEpsilon, best_optimizing_index] = OptimThreshold(ycval, p)
% Find the best threshold (epsilon) to use for selecting anomalies
% ycval = responses Y=1 if anomaly, 0 otherwise
% p = density;

bestEpsilon = 0;
best_optimizing_index = 0;
stepsize = (max(p) - min(p)) / 10000;

for epsilon = min(p):stepsize:max(p)

    predictions = p < epsilon;
    tp = sum((predictions == 1) & (ycval == 1));
    fp = sum((predictions == 1) & (ycval == 0));
    fn = sum((predictions == 0) & (ycval == 1));
    tn = sum((predictions == 0) & (ycval == 0));
    accuracy = (tp+tn)/(tp+tn+fp+fn);
    prec = tp / (tp + fp);
    rec = tp / (tp + fn);
    F1 = 2 * prec * rec / (prec + rec);
    % optimizing_index = 100*tp - 75*fp;
    % optimizing_index = accuracy; % all 0
    % optimizing_index = rec; % all 1
    optimizing_index = accuracy * rec / (accuracy + rec);
    % optimizing_index = rec - accuracy;
    
    if optimizing_index> best_optimizing_index
        best_optimizing_index = optimizing_index;
        bestEpsilon = epsilon;
    end
end
end


