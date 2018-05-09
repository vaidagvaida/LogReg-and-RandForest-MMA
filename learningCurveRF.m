function [number_trees, error_train, error_val] = learningCurveRF(X, Y, maxTrees,incrementTrees,MinSLeafSize, MaxFeatures)

%learningCurveRF generates training, and out of bag errors, for random
%forrest algorithms with increasing number of trees.

%X is training data
%y is class labels
%maxTrees is the maximum number of trees to be considered.
%incrementTrees is the increment of number of trees in consideration
%MinSLeafSize is minimum number of samples per leaf
%MaxFeatures is maximum number of features per split

% Return error values correctly
error_train = zeros(maxTrees/incrementTrees, 1);
error_val   = zeros(maxTrees/incrementTrees, 1);
number_trees = zeros(maxTrees/incrementTrees, 1);
indx=1; %Create indexing for data. 

% Loop over the training examples and calculate errors.
for i = incrementTrees:incrementTrees:maxTrees+1
    
    fprintf('Learning... number of trees: %d\n', i) 
    % Check if we do not get only one class in our subset of data, in which
    % case C has to be equal 1.
   
   % training a single model (last modelT will be the fully trained one)
   randomForest =  TreeBagger(i,X,Y,'Method','classification',...
    'OOBPrediction','on','MinLeafSize',MinSLeafSize,...
    'NumPredictorstoSample', MaxFeatures);
   oobErr = mean(oobError(randomForest));
   err = mean(error(randomForest,X,Y));
   
   error_train(indx)=err; % Store the size of a training data
   error_val(indx)=oobErr;
   number_trees(indx)=i;
   indx=indx+1; % Move of an index, of a data storage.
   
   fprintf('Learning finished. Out of bag error: %d\n', oobErr) 

   
 
end
fprintf('DONE')
end