function [iterations, accuracy_train, accuracy_test] = learningCurveLR(X, Y,Xtest, Ytest,w0, maxIterations,incrementIterations,learnrate)

%learningCurveRF generates training, and out of bag errors, for random
%forrest algorithms with increasing number of trees.

%X is training data
%y is class labels
%maxIterations is the maximum number of logistic regression iterations
%incrementIterations is the increment of number of iterations in consideration
%lambda is logistic regression regularization parameter

% Return error values correctly
accuracy_train = zeros(maxIterations/incrementIterations, 1);
accuracy_test   = zeros(maxIterations/incrementIterations, 1);
iterations = zeros(maxIterations/incrementIterations, 1);
indx=1; %Create indexing for data. 

% Loop over the training examples and calculate errors.
for i = incrementIterations:incrementIterations:maxIterations+1
    
    fprintf('Learning... number of logistic iterations: %d\n', i) 
    % Check if we do not get only one class in our subset of data, in which
    % case C has to be equal 1.
   
   % training a single model (last modelT will be the fully trained one)
   weight = logisticRegressionWeights( X, Y, w0, i, learnrate);  
   [pred_probs_train,pred_classes_train] = logisticRegressionClassify( X, weight );   
   [pred_probs_test,pred_classes_test] = logisticRegressionClassify( Xtest, weight );

   correct_train = (pred_classes_train == Y);
   accuracy_train(indx)= sum(correct_train)/length(correct_train);
   
   correct_test = (pred_classes_test == Ytest);
   accuracy_test(indx)= sum(correct_test)/length(correct_test);
   
   iterations(indx)=i;
   indx=indx+1; % Move of an index, of a data storage.
   
   fprintf('Learning finished. Testing accuracy: %d\n', accuracy_test) 

   
 
end
fprintf('DONE')
end