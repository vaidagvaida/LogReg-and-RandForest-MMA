%% 6. Confusion matrices, precision , recall, f1

% Random forests

valTarg_cat = categorical(valTarg);
RF_predictions_cat=categorical(RF_predictions);

% Feature importances
feature_importances=randomForestBest.OOBPermutedVarDeltaError;

% Confusion matrices random forest
RF_predictions_d = str2double(RF_predictions); % Convert to double
plotconfusion(transpose(valTarg),transpose(RF_predictions_d), 'Random forest confusion matrix');

% Random forest accuracy
err_RF = mean(error(randomForestBest,valData,valTarg)); % Testing accuracy
validationAccuracy_RF=1-err_RF;

err_RF2 = mean(error(randomForestBest,trainData,trainTarg)); % Training accuracy
trainingAccuracy_RF=1-err_RF2;

% Logistic regression

LR_predictions_cat=categorical(LR_predictions);

C_LR = confusionmat(valTarg_cat, LR_predictions_cat); % Confusion matrices

% Confusion matrix logistic regression
plotconfusion(transpose(valTarg),transpose(LR_predictions), 'Logistic regression confusion matrix');

% Logistic regression accuracy
correctPredictions = (LR_predictions == valTarg); % Testing accuacy
validationAccuracy_LR = sum(correctPredictions)/length(correctPredictions);

[LR_scores2, LR_predictions2] = logisticRegressionClassify( trainData, weights_best );

correctPredictions2 = (LR_predictions2 == trainTarg); % Training accuracy
trainingaccuracy_LR = sum(correctPredictions2)/length(correctPredictions2);

% Bookmaker accuracy
% Derived from probability estimates with threshold of 0.5 to seperate
% classes
Bookmaker_prob_train=trainData(:,1:1);
Bookmaker_prob_test=valData(:,1:1);
Bookmaker_class_train=zeros(length(trainData), 1);
Bookmaker_class_test=zeros(length(valData),1);

for x = 1  : length(Bookmaker_prob_train)
   if Bookmaker_prob_train(x)>0.5
     Bookmaker_class_train(x) = 1;
   else
     Bookmaker_class_train(x) = 0;
   end
end

for x = 1  : length(Bookmaker_prob_test)
   if Bookmaker_prob_test(x)>0.5
     Bookmaker_class_test(x) = 1;
   else
     Bookmaker_class_test(x) = 0;
   end
end

correctPredictions = (Bookmaker_class_train == trainTarg); % Training accuacy
trainingAccuracy_book = sum(correctPredictions)/length(correctPredictions);

correctPredictions = (Bookmaker_class_test == valTarg); % Testing accuacy
validationAccuracy_book = sum(correctPredictions)/length(correctPredictions);

%% 7. Get probabilities, draw reliability curves 

RF_probabilities=RF_scores(:,2:2); % Random forrest probability estimates
LR_probabilities=LR_scores; % Logistic regression probability estimates


[fig2, targets_means_RF, predictions_means_LR] = plotreliability(valTarg, LR_probabilities, 'false'); % Reliability curves
[fig2, targets_means_LR, predictions_means_RF] = plotreliability(valTarg, RF_probabilities,'false'); % Reliability curves
[fig2, targets_means_Implied, predictions_means_Implied] = plotreliability(valTarg, Bookmaker_prob_test, 'false'); % Reliability curves

bs_RF = brier_score(RF_probabilities,valTarg); % Brier score RF
bs_LR = brier_score(LR_probabilities,valTarg); % Brier score LR
bs_Implied = brier_score(Bookmaker_prob_test,valTarg); % Brier score Bookm.


%% Plot figure

fig = figure('Visible', 'on');
axes1 = axes('Parent', fig);

box(axes1, 'on')
hold(axes1, 'all')

plot(axes1, [0, 1], [0, 1], 'Color', 'black')
plot(axes1, predictions_means_RF, targets_means_RF, 'Color', 'blue', ...
     'Marker', 'o')
plot(axes1, predictions_means_LR, targets_means_LR, 'Color', 'red', ...
     'Marker', 'o')
plot(axes1, predictions_means_Implied, targets_means_Implied, 'Color', 'green', ...
     'Marker', 'o')

legend(['Perfectly calibrated probabilities'],['Random Forest brier score=' num2str(bs_RF)],['Logistic regression brier score=' num2str(bs_LR)],['Implied probability brier score=' num2str(bs_Implied)]);
xlabel(axes1, 'Mean Predicted Value')
ylabel(axes1, 'Fraction of Positives')
title(axes1, 'Reliability Diagram')

saveas(fig,'reliability_curves.jpg')
hold(axes1, 'off')