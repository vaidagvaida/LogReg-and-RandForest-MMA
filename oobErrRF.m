% Objective function

function oobErr = oobErrRF(params,X,Y)
%oobErrRF Trains random forest and estimates out-of-bag error
%   oobErr trains a random forest of 100 classification trees using the
%   predictor data in X and the parameter specification in params, and then
%   returns the out-of-bag error based on the mean. X is a table
%   and params is an array of OptimizableVariable objects corresponding to
%   the minimum leaf size and number of predictors to sample at each node.
randomForest = TreeBagger(100,X,Y,'Method','classification',...
    'OOBPrediction','on','MinLeafSize',params.MinSLeafSize,...
    'NumPredictorstoSample',params.MaxFeatures);
oobErr = mean(oobError(randomForest));
% oobErr = oobQuantileError(randomForest);
end

% MdlCART = TreeBagger(1000,X,Y,'Method','classification','Surrogate','on',...
%     'OOBPredictorImportance','on');