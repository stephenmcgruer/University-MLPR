function [ ys, variances ] = n_fold_linear_regression( n, x, y )
%N_FOLD_LINEAR_REGRESSION Computes n-fold cross validation linear
%regression on x and y.

% Netlab is needed for glm, etc.
path(path, 'netlab');

% Split the data into n pieces.
split_size = length(x) / n;
x_split = mat2cell(x, repmat(split_size, n, 1), size(x, 2));
y_split = mat2cell(y, repmat(split_size, n, 1), size(y, 2));

% The number of inputs to the linear regressor.
number_inputs = size(x, 2);

% Perform the n-fold cross validation linear regression.
y_cell = cell(n);
vs = zeros(n, 1);
for i = 1 : n
    x_validation = x_split;
    x_validation(i) = [];
    x_validation = cell2mat(x_validation);
    
    y_validation = y_split;
    y_validation(i) = [];
    y_validation = cell2mat(y_validation);
    
    % Create a linear regression network.
    net = glm(number_inputs, 1, 'linear');
    
    % Train the weights using x_validation. This is roughly equivalent to:
    % w = [ones(size(x_validation, 1), 1) x_validation] \ y_validation;
    options = zeros(1, 18);
    net = glmtrain(net, options, x_validation, y_validation);
    
    % Now pass the validation inputs back through the trained weights to
    % estimate the variance of error.
    % As the mean is 0, the variance is equivalent to the mean squared
    % error.
    w_phi_x = glmfwd(net, x_validation);
    vs(i) = mean((y_validation - w_phi_x).^2);
    
    % Finally, pass x_test into the linear regressor to compute the
    % values for y. This is roughly equivalent to:
    % y_cell{i} = (w' * [ones(size(x_test, 1), 1) x_test]')';
    x_test = cell2mat(x_split(i));
    y_cell{i} = glmfwd(net, x_test);
end

ys = cell2mat(y_cell);

% Replicate the vs to match them row-by-row to the ys.
variances = reshape(repmat(vs', split_size, 1), [], 1);

end

