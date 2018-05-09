%0. Correlation matrix

RHO = corr(trainDataFull, 'rows', 'complete'); %Correlation matrix
h = heatmap(RHO); % Correlation heatmap

% 4th and 5th column are very highly correlated, therefore one to be
% excluded (TotalWins_diff_Scaled excluded)
