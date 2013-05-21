function [ parameters, priors ] = train_naivebayes( x, y )
%TRAIN_NAIVEBAYES Estimates the parameters of a multinomial naive bayes.
%   Assumes there are 64 classes and 64 values. Uses add-one smoothing
%   for both the priors and class conditionals.
%
%   x is a n-by-f matrix where each row is a training input and each
%   column encodes a feature.
%
%   y is a n-by-1 vector giving the correct class for each training input.
%   It is assumed that each class is numeric.
%
%   The returned parameters variable is a 64-by-64-by-f matrix of the
%   parameters for each possible feature value for each class and feature.
%   The priors are also returned, in a 64-by-1 array.

parameters = zeros(64, 64, size(x, 2));
priors = zeros(64, 1);

% For each class, find the training rows for that class, and calculate the
% parameters and priors.
for c = 1 : 64
    x_c = x;
    
    % Remove the test rows with class ~= c.
    other_classes = y ~= (c - 1);
    x_c(other_classes, :) = [];
    
    num_training_inputs = length(x_c);
    
    priors(c) = num_training_inputs + 1;
    
    % For each possible value, compute the counts of how many features
    % have that value (for class c).
    for v = 1 : 64
        counts = sum(x_c == (v - 1)) + 1;
        parameters(c, v, :) = counts / (num_training_inputs + 64);
    end
end

% Divide the priors through to get the correct ratios.
priors = priors ./ (length(x) + 64);

end

