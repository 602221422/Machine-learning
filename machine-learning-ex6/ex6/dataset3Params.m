function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
vec=[0.01,0.03,0.1,0.3,1.3,10,30]';
C=0.01;
sigma=0.01;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
%SVMPREDICT使用经过训练的SVM模型返回一个预测向量。
% pred = SVMPREDICT(model, X)使用经过训练的SVM模型(svmTrain)返回一个预测向量。X是一个mxn矩阵，每个例子都是一行。model是svmTrain返回的svm模型。
%pred是一个mx1列的{0,1}值的预测,返回的是经过训练的SVM模型拟合xval数据生成的0，1预测值
predictions=svmPredict(model,Xval);
%看经过训练后的model与交叉验证训练集的数据的相吻合度
meanMin=mean(double(predictions~=yval));
C_optimal=C;
sigma_optimal=sigma;
for i=1:length(vec)  
    for j=1:length(vec)
        C=vec(i);
        sigma=vec(j);
        model= svmTrain(X, y, C, @(x1, x2) ...
               gaussianKernel(x1, x2, sigma)); 
        predictions=svmPredict(model,Xval);
        meantest=mean(double(predictions~=yval));
        if(meanMin >= meantest)
              meanMin=meantest;
              C_optimal=C;
              sigma_optimal=sigma;
        end
    end
end
C=C_optimal;
sigma=sigma_optimal;      
% ======================================================================

end
