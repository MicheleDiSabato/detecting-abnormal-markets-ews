clc
clear 
close all

% Load data 
load('C:\Users\Michele\Desktop\detecting-abnormal-markets-ews\EWS.')


%% FEATURE STAZIONARIZATION 

% Always positive variables   => log-differences (log-returns)
Indices_Currencies = [XAUBGNL BDIY CRY Cl1 DXY EMUSTRUU GBP JPY LF94TRUU...
                   LF98TRUU LG30TRUU LMBITR LP01TREU...
                   LUACTRUU LUMSTRUU MXBR MXCN MXEU MXIN MXJP MXRU MXUS VIX];

% Possibly negative variables => first differences
InterestRates = [EONIA GTDEM10Y GTDEM2Y GTDEM30Y GTGBP20Y GTGBP2Y GTGBP30Y...
                 GTITL10YR GTITL2YR GTITL30YR GTJPY10YR GTJPY2YR...
                 GTJPY30YR US0001M USGG3M USGG2YR GT10 USGG30YR];
             
% Create dataset 
X = [diff(log(Indices_Currencies)) ECSURPUS(2:end) diff(InterestRates) Y(1:end-1)]; 
X = [X(2:end,:) Y(1:end-2)];
% Response 
Response = Y(3:end);      

% Time window 
Days = Data(2:end);       


%% FEATURE SELECTION 

selected_cols = [2 3 25 27 30 16 18 22 40 23];
% X = X(:,selected_cols);


%% SPLITTING 

% N = size(X(:,1));
% train_perc = 0.8;
% split = round(train_perc*N);
% 
% X_train = X(1:split,:);
% X_test  = X(split+1:end,:);
% 
% Y_train = Response(1:split,:);
% Y_test  = Response(split+1:end,:);

nObs = length(Response); % #data
nObsNorm = sum(Response == 0); % #normal data points
nObsAbNorm = nObs - nObsNorm; % #abnormal data points
percentage_normal = 0.8;
altra_percentuale = (1-percentage_normal)/2;
nObsTrain = round(percentage_normal*nObsNorm); % #length training set
nObsCVNorm = round(altra_percentuale*nObsNorm); % #length normal portion of CV set
nObsCVabNorm = nObsCVNorm; % #length abnormal portion of CV set
nObsTest_abNorm = nObsAbNorm - nObsCVabNorm; % #length abnormal portion of Train set

% Reshuffling the sample
idxPermutation = randperm(nObs);
X = X(idxPermutation,selected_cols);
Response = Response(idxPermutation);

% dividing normal/abnormal
Xnormal = X(Response == 0,:);
Xabnormal = X(Response == 1,:);
Yabnormal = Response(Response == 1,:);

X_train = Xnormal(1:nObsTrain,:);
XCV = [Xnormal(nObsTrain+1:nObsTrain+1+nObsCVNorm,:); Xabnormal(1:nObsCVabNorm,:)];
non_so_come_chiamarlo = Xabnormal(nObsCVabNorm+1:1:end,:);
X_test = [Xnormal(nObsTrain+1+nObsCVNorm+1:end,:); non_so_come_chiamarlo];
L1 = size(Xnormal(nObsTrain+1+nObsCVNorm+1:end,:), 1);
L2 = size(non_so_come_chiamarlo, 1);

yCV = zeros(length(XCV),1);
yCV(end-nObsCVabNorm+1:end) = Yabnormal(1:nObsCVabNorm);
Y_test = zeros(length(X_test),1);
% Y_test(end-nObsTest_abNorm+1:end) = Yabnormal(nObsCVabNorm+1:end);
Y_test(end-L2:end) = 1;


%% DATA RESCALING 

for i = 1:numel(selected_cols)
   X_test(:,i) = rescale(X_test(:,i),min(X_train(:,i)),max(X_train(:,i))); 
end

for i = 1:numel(selected_cols)
   XCV(:,i) = rescale(XCV(:,i),min(X_train(:,i)),max(X_train(:,i))); 
end

for i = 1:numel(selected_cols)
   X_train(:,i) = rescale(X_train(:,i)); 
end


%% SUBSAMPLING 
% 
% X_train_0 = X_train(Y_train==0,:);
% X_train_1 = X_train(Y_train==1,:);
% 
% indexes = randi(size(X_train_1,1),size(X_train_1,1),1);
% 
% X_train_balanced = [X_train_0(indexes,:);X_train_1];
% Y_train_balanced = [zeros(size(X_train_1,1),1);ones(size(X_train_1,1),1)];

%% COPULAS
colSelector = [2 3 25 27 30 16 18 22 40 23];
[nSample, nFeatures] = size(X_train(:,:));
uTrain = zeros(nSample, nFeatures);

for i = 1:nFeatures
    uTrain(:,i) = ksdensity(X_train(:,i), X_train(:,i), 'function', 'cdf');
end

[rhohat0] = copulafit('Gaussian', uTrain); % fit Copula t

%% CV
[nSample, nFeatures] = size(XCV(:,:));
uCV = zeros(nSample, nFeatures);

for i = 1:nFeatures
    uCV(:,i) = ksdensity(XCV(:,i), XCV(:,i), 'function', 'cdf');
end

p = copulapdf('Gaussian',uCV,rhohat0); % parameters estimated on the train set

[bestEpsilon, best_optimizing_index] = OptimThreshold(yCV, p) % find the optimal hyperparameter

%% TEST

[nSample, nFeatures] = size(X_test(:,:));
uTest = zeros(nSample, nFeatures);

for i = 1:nFeatures
    uTest(:,i) = ksdensity(X_test(:,i), X_test(:,i), 'function', 'cdf');
end

p = copulapdf('Gaussian',uTest,rhohat0);

predictions = p < bestEpsilon;
tp = sum((predictions == 1) & (Y_test == 1))
fp = sum((predictions == 1) & (Y_test == 0))
fn = sum((predictions == 0) & (Y_test == 1))
tn = sum((predictions == 0) & (Y_test == 0))

acc = (tp+tn)/(tp+fp+tn+fn)
prec = tp / (tp + fp)
rec = tp / (tp + fn)
F1 = 2 * prec * rec / (prec + rec)
C = confusionmat(predictions,logical(Y_test))

