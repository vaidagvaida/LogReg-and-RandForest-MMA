
%% Creating weights for logistic regression 

function [w] = logisticRegressionWeights( Predictors_Train, Target_Train, w0, maxIter, learningRate )

    [nSamples, nFeature] = size(Predictors_Train);
    w = w0; % weights randomly generated from optimization code
    precost = 0; % cost before any adjustments were done 
    for j = 1:maxIter %adjust weights n times (maxiterations set in optimization code)
        theta = zeros(nFeature + 1,1);  % will hold theta value 
        for k = 1:nSamples
            theta = theta + (sigmoid([1.0 Predictors_Train(k,:)] * w) - Target_Train(k)) * [1.0 Predictors_Train(k,:)]'; %% calculating gradient descent 
        end
        w = w - learningRate * theta; % adjusting weights using gradient descent
        cost = CostFunc(Predictors_Train, Target_Train, w); % calculating cost funtion, defined in CostFunc.m
        if j~=0 && abs(cost - precost) / cost <= 0.001 % stopping criteria set
            break;
        end
        precost = cost;
    end

end
