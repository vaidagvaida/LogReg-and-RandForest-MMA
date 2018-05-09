%% Logistic regression cost function

function [J] = CostFunc( predictors_train, target_train, w )

    [nSamples, nFeature] = size(predictors_train); %getting size of data
    theta = 0.0; %theta  set to 0
    for m = 1:nSamples
        hx = sigmoid([1.0 predictors_train(m,:)] * w); % getting sigmpid (defined in sigmoid.m)
        if target_train(m) == 0  %% in this section, a cost function will be calculated with an application of gradient descent
            theta = theta + log(1 - hx);%%for losing class (0) has to be adjusted because otherwise it will show that cost is very large (as cost function is modelled around winning class and the assignment to a losing class would make the cost seem to be "through the roof")     
        else
          theta = theta + log(hx); %% for winning class (1)   
        end
    end
    J = theta / (-nSamples); %% computes the cost of using using theta parameter in logistic regression

end