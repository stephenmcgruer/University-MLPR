function [ ys, variances ] = n_fold_rbf( n, x, y )
%N_FOLD_LINEAR_REGRESSION Computes n-fold cross validation rbf
%regression on x and y.

% Netlab is needed for rbf, etc.
path(path, 'netlab');

% Split the data into n pieces.
split_size = length(x) / n;
x_split = mat2cell(x, repmat(split_size, n, 1), size(x, 2));
y_split = mat2cell(y, repmat(split_size, n, 1), size(y, 2));

% The number of inputs to the rbf network.
number_inputs = size(x, 2);

% Perform the n-fold cross validation rbf regression.
y_cell = cell(n);
vs = zeros(n, 1);
for i = 1 : n
    x_validation = x_split;
    x_validation(i) = [];
    x_validation = cell2mat(x_validation);
    
    y_validation = y_split;
    y_validation(i) = [];
    y_validation = cell2mat(y_validation);
    
    % Create a rbf network.
    net = rbf(number_inputs, 5, 1, 'gaussian');
    
    % Train the weights using x_validation.
    options = zeros(1, 18);
    net = rbftrain(net, options, x_validation, y_validation);
    
    % Now pass the validation inputs back through the network to
    % estimate the variance of error.
    y_out = rbffwd(net, x_validation);
    vs(i) = mean((y_validation - y_out).^2);
    
    % Finally, pass x_test into the rbf network to compute the values
    % for y.
    x_test = cell2mat(x_split(i));
    y_cell{i} = rbffwd(net, x_test);
end

ys = cell2mat(y_cell);

% Replicate the vs to match them row-by-row to the ys.
variances = reshape(repmat(vs', split_size, 1), [], 1);

end

