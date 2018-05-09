%1. DATA PREP

data = xlsread('Fights_fighters_odds_engineered.xls'); % Loading data

data(:,[6,7,9,12])=[]; % Delete weight classes (nan columns);...
%fighter 2 implied prob (not needed); and total wins (multicolinear with
%total fights)
data=data(:, 5:end-4);
rows_to_be_removed = any( isnan( data ), 2 ); % Remove NaNs...
data(rows_to_be_removed,:)=[];

trainDataFull=data(:, 2:end); % Read feature data matrix
trainTargFull=data(:,1:1); % Read target data matrix


[trainInd,valInd] = dividerand(length(trainDataFull),0.8,0.2); % Creating indices for splitting

trainData = trainDataFull(trainInd, :); % Splits the data according to the generated index
trainTarg=trainTargFull(trainInd, :);
valData=trainDataFull(valInd, :);
valTarg=trainTargFull(valInd, :);