function [bestEpsilon,bestValue] = OptimThreshold(ycval, p)
% Find the best threshold (epsilon) to use for selecting anomalies
% ycval = responses Y=1 if anomaly, 0 otherwise
% p = density;

% Initialization
bestEpsilon = 0;
bestValue   = 0;
rec_        = 0;
prec_       = 0;

stepsize = (max(p) - min(p)) / 1000;

for epsilon = min(p):stepsize:max(p)

    predictions = p < epsilon;
    
    tp = sum((predictions == 1) & (ycval == 1));
    fp = sum((predictions == 1) & (ycval == 0));
    fn = sum((predictions == 0) & (ycval == 1));
    tn = sum((predictions == 0) & (ycval == 0));
    
    prec = tp / (tp + fp);
    rec  = tp / (tp + fn);
    F1   = 2 * prec * rec / (prec + rec);
    
    % POSSIBLE CHOICES 
    % optimize = (1/2)*prec + (1/2)*rec;          % lin comb of prec & rec 
    % optimize = (tp + tn) /(tp + tn + fn + fp);  % accuracy
    
    optimize = rec;     % recall 
    
    if optimize > bestValue
        
        % Current values 
        prec_ = prec;
        rec_  = rec;
        
        % Best values 
        bestValue   = optimize;
        bestEpsilon = epsilon;
    end
end

end


