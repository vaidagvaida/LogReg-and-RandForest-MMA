%% 4. Random forest training full model on best parameters 

t1=datetime('now');
randomForestBest = TreeBagger(250,trainData,trainTarg,'Method','classification',...
    'MinLeafSize',results.XAtMinObjective.MinSLeafSize,...
    'NumPredictorstoSample',results.XAtMinObjective.MaxFeatures, 'OOBPredictorImportance','on');
[RF_predictions,RF_scores] = predict(randomForestBest,valData);
t2=datetime('now');
durFullTrainRF=t2-t1;

%% 5. Logistic regression full model on best parameters

[nsamples, nfeatures] = size(trainData);
w0 = rand(nfeatures + 1, 1); % Intialize random weights


t1=datetime('now');
weights_best = logisticRegressionWeights( trainData, trainTarg, w0, 50, bestLogistic.learn_rate);
[LR_scores, LR_predictions] = logisticRegressionClassify( valData, weights_best );
t2=datetime('now');
durrFullTrainLR=t2-t1;







