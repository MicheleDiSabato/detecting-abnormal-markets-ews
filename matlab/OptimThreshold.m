function [bestEpsilon bestF1] = OptimThreshold(ycval, p)
% Find the best threshold (epsilon) to use for selecting anomalies
% ycval = responses Y=1 if anomaly, 0 otherwise
% p = density;

bestEpsilon = 0;
bestF1 = 0;
stepsize = (max(p) - min(p)) / 1000;

for epsilon = min(p):stepsize:max(p)

    predictions = p < epsilon;
    tp = sum((predictions == 1) & (ycval == 1));
    fp = sum((predictions == 1) & (ycval == 0));
    fn = sum((predictions == 0) & (ycval == 1));
    prec = tp / (tp + fp);
    rec = tp / (tp + fn);
    F1 = 2 * prec * rec / (prec + rec);

    if F1 > bestF1
        bestF1 = F1;
        bestEpsilon = epsilon;
    end
    
end

end


