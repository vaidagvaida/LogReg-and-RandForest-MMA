%% 2. Random forest hyperparameter optimization

% Prepare variables for Baysian optimization
%GRIDS%
MaxFeatures = optimizableVariable('MaxFeatures', [1,9],'Type','integer');
MinSLeafSize = optimizableVariable('MinSLeafSize', [1,11],'Type','integer');
hyperparametersRF=[MaxFeatures, MinSLeafSize]; 
%%% GRIDS SET %%%

%Optimize bayesian objective function (results)
t1=datetime('now');
results = bayesopt(@(params)oobErrRF(params,trainData,trainTarg),hyperparametersRF,...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',1);
t2=datetime('now');
durByesOpt=t2-t1;


%% 3. Random forest learning curves (number of trees optimization)

%Learning curves (training data) %

t1=datetime('now');
[number_trees, error_train, error_val] = learningCurveRF(trainData, trainTarg, 500,25,...
    results.XAtMinObjective.MaxFeatures,results.XAtMinObjective.MinSLeafSize);
t2=datetime('now');
durLearnCurveRF=t2-t1;

train_accuracy=1-error_train; % training accuracy
oob_accuracy=1-error_val; % out of bag accuracy

figure1=figure;
plot(number_trees,train_accuracy,number_trees,oob_accuracy)
title('Learning curves Random Forest')
xlabel('Number of trees')
ylabel('Classification accuracy')
legend('Training data','Out of bag data')
saveas(figure1,'learningcurvesRF.jpg')


%% 4. Logistic regression lambda optimization

gridLearn = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.3,0.5,0.7,0.9]; % Learning rate grid specified

% Split training data for logistic optimization testing (80/20 split)
% (Note: split is random and not say first 20% as test data and the resr as
% train data) 

% Split training data for logistic optimization testing (80/20 split)
[train2Ind,val2Ind] = dividerand(length(trainData),0.8,0.2); % Creating indices for splitting
train2Data = trainData(train2Ind, :); % Splits the data according to the generated index
train2Targ=trainTarg(train2Ind, :);
val2Data=trainData(val2Ind, :);
val2Targ=trainTarg(val2Ind, :);

[nsamples, nfeatures] = size(train2Data);
w0 = rand(nfeatures + 1, 1); % Initialize random weights


% Struct for best logistic model
bestLogistic = struct('LogisticModel', NaN, 'learn_rate', NaN, 'Accuracy', 0); %Stores Best Logistic model

% Grid search for best RBF_Sigma/Kernel Scale 
count=0;
t1=datetime('now');
for learn_rate = gridLearn
    % Grid search for best Lambda (regularization parameter)
    t3=datetime('now');
    fprintf('TESTING PARAMETERS: learning rate:%d', learn_rate)
    weight = logisticRegressionWeights( train2Data, train2Targ, w0, 1000, learn_rate);    
    [pred_probs,pred_classes] = logisticRegressionClassify( val2Data, weight );
    correctPredictions = (pred_classes == val2Targ);
    validationAccuracy = sum(correctPredictions)/length(correctPredictions);
    fprintf('TESTING ACCURACY: %d\n', validationAccuracy);
    t4=datetime('now');
    fprintf('...TIME COST OF TRAINING: %s\n', t4-t3)
    fprintf('----------------------------------------------\n')
    if validationAccuracy > bestLogistic.Accuracy
            bestLogistic.Accuracy = validationAccuracy;
            bestLogistic.learn_rate=learn_rate;
            bestLogistic.LogisticModel=weight;
    end
end
t2=datetime('now');
durGridSearch=t2-t1;

%% 5. Logistic regression learning curves

maxIterations=100;
incrementIterations=5;

t1=datetime('now');
[iterations, accuracy_train, accuracy_test] = learningCurveLR(train2Data, train2Targ,val2Data,val2Targ,w0, maxIterations,incrementIterations,bestLogistic.learn_rate);
t2=datetime('now');
durLearnCurveLR=t2-t1;

figure2=figure;
plot(iterations,accuracy_train,iterations,accuracy_test)
title('Learning curves Logistic regression')
xlabel('Number of iterations')
ylabel('Classification accuracy')
legend('Training data','Testing data')
saveas(figure2,'learningcurvesLR.jpg')



