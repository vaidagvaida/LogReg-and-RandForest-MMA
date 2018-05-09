%% Building LR classifier

function [pred, res] = logisticRegressionClassify( Predictors_test, w )
   
    nTest = size(Predictors_test,1); % getting the size of data
    res = zeros(nTest,1); % will hold responses (classes)
	pred =zeros(nTest,1); % will hold sigmoid value
    for i = 1:nTest 
        sigm = sigmoid([1.0 Predictors_test(i,:)] * w); %% calculatimg the sigmoid with test predictors and weights ,  this will be used to clasify the results. 
		pred(i)=sigm
        if sigm >= 0.5 % setting decision boundry 
            res(i) = 1; %win fight
        else
            res(i) = 0; %lost fight
        end
    end

end