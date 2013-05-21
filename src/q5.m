%% Adding 'i' to Linear Regression - Setup
disp('5 (Linear Regression with i).');

% Tidy up any previous state.
clear all
close all

% The base of the file locations.
base = '/afs/inf.ed.ac.uk/group/teaching/mlprdata/challengedata/';

% If the user is not on dice, they have to enter the location of the file
% manually.
try
    load(strcat(base, 'imdata.mat'));
catch %#ok<CTCH>
    disp('Unable to find imdata.mat on afs. Please enter location: ');
    location = input('? ', 's');
    load(location);
end

% Convert to double and remove un-needed columns. Renamed to avoid
% conflicts later on if xmtestdata.mat is loaded. Append the i values
% to x_data.
x_data = stdz([double(x(:, [end, end - 34, end - 35])), i]);
y_data = double(y);

% Free up some memory.
clear 'i' 'x' 'y' 'base' 'location';

disp('Data loaded.');

%% Adding 'i' to Linear Regression - Main

[ys, variances] = n_fold_linear_regression(4, x_data, y_data);

disp('Linear regression values calculated.');

% To calculate the output probabilities, create a gaussian centered around
% each output y value, with a variance drawn from the respective row in 
% 'variances'.
probs = zeros(length(ys), 64);
for i = 1 : length(ys)
    mu = ys(i);
    sigma = sqrt(variances(i));
    
    for c = 1 : 64
        probs(i, c) = gauss_distribution(c - 1, mu, sigma);
    end
    
    % Normalize the probabilities.
    total = sum(probs(i, :));
    for c = 1 : 64
        probs(i, c) = probs(i, c) / total;
    end
end

disp('Class probabilities calculated.');

perplex = perplexity(probs, y_data, 0);

disp('Perplexity calculated.');
perplex

%% Adding 'x(:, end - 33)' to Linear Regression - Setup
disp('5 (Linear Regression with x(:, end - 33)).');

% Tidy up any previous state.
clear all
close all

% The base of the file locations.
base = '/afs/inf.ed.ac.uk/group/teaching/mlprdata/challengedata/';

% If the user is not on dice, they have to enter the location of the file
% manually.
try
    load(strcat(base, 'imdata.mat'));
catch %#ok<CTCH>
    disp('Unable to find imdata.mat on afs. Please enter location: ');
    location = input('? ', 's');
    load(location);
end

% Convert to double and remove un-needed columns. Renamed to avoid
% conflicts later on if xmtestdata.mat is loaded.
x_data = double(x(:, [end, end - 33, end - 34, end - 35]));
y_data = double(y);

% Free up some memory.
clear 'i' 'x' 'y' 'base' 'location';

%% Adding 'x(:, end - 33)' to Linear Regression - Main

[ys, variances] = n_fold_linear_regression(4, x_data, y_data);

disp('Linear regression values calculated.');

% To calculate the output probabilities, create a gaussian centered around
% each output y value, with a variance drawn from the respective row in 
% 'variances'.
probs = zeros(length(ys), 64);
for i = 1 : length(ys)
    mu = ys(i);
    sigma = sqrt(variances(i));
    
    for c = 1 : 64
        probs(i, c) = gauss_distribution(c - 1, mu, sigma);
    end
    
    % Normalize the probabilities.
    total = sum(probs(i, :));
    for c = 1 : 64
        probs(i, c) = probs(i, c) / total;
    end
end

disp('Class probabilities calculated.');

perplex = perplexity(probs, y_data, 0);

disp('Perplexity calculated.');
perplex

%% Adding 'x(:, end - 33)' to Naive Bayes - Setup
disp('5 (Naive Bayes with x(:, end - 33)).');

% Tidy up any previous state.
clear all
close all

% The base of the file locations.
base = '/afs/inf.ed.ac.uk/group/teaching/mlprdata/challengedata/';

% If the user is not on dice, they have to enter the location of the file
% manually.
try
    load(strcat(base, 'imdata.mat'));
catch %#ok<CTCH>
    disp('Unable to find imdata.mat on afs. Please enter location: ');
    location = input('? ', 's');
    load(location);
end

% Convert to double and remove un-needed columns. Renamed to avoid
% conflicts later on if xmtestdata.mat is loaded.
x_data = double(x(:, [end, end - 33, end - 34, end - 35]));
y_data = double(y);

% Free up some memory.
clear 'i' 'x' 'y' 'base' 'location';

% Split the data into 4 sets for cross validation.
split_length = length(x_data) / 4;
x_split = mat2cell(x_data, repmat(split_length, 4, 1), size(x_data, 2));
y_split = mat2cell(y_data, repmat(split_length, 4, 1), size(y_data, 2));

disp('Data loaded.');

%% Adding 'x(:, end - 33)' to Naive Bayes - Main

% N-fold cross validation.
prob_out = cell(length(x_split), 1);
for i = 1 : length(x_split)
    x_validation = x_split;
    x_validation(i) = [];
    x_validation = cell2mat(x_validation);
    
    y_validation = y_split;
    y_validation(i) = [];
    y_validation = cell2mat(y_validation);
    
    x_test = cell2mat(x_split(i));
    y_test = cell2mat(y_split(i));
    
    [outputs, priors] = train_naivebayes(x_validation, y_validation);
    
    prob_out{i} = test_naivebayes(x_test, outputs, priors);
end

% Calculate the perplexity.
perplex = perplexity(cell2mat(prob_out), y_data, 1);

disp('4-fold cross validation perplexity:');
perplex

%% Using a Radial Basis Function Network - Setup
disp('5 (Radial Basis Function Network).');

% Tidy up any previous state.
clear all
close all

% The base of the file locations.
base = '/afs/inf.ed.ac.uk/group/teaching/mlprdata/challengedata/';

% If the user is not on dice, they have to enter the location of the file
% manually.
try
    load(strcat(base, 'imdata.mat'));
catch %#ok<CTCH>
    disp('Unable to find imdata.mat on afs. Please enter location: ');
    location = input('? ', 's');
    load(location);
end

% Convert to double and remove un-needed columns. Renamed to avoid
% conflicts later on if xmtestdata.mat is loaded.
x_data = double(x(:, [end, end - 33, end - 34, end - 35]));
y_data = double(y);

% Free up some memory.
clear 'i' 'x' 'y' 'base' 'location';

disp('Data loaded.');

%% Using a Radial Basis Function Network - Main

[ys, variances] = n_fold_rbf(4, x_data, y_data);

disp('RBF values calculated.');

% To calculate the output probabilities, create a gaussian centered around
% each output y value, with a variance drawn from the respective row in 
% 'variances'.
probs = zeros(length(ys), 64);
for i = 1 : length(ys)
    mu = ys(i);
    sigma = sqrt(variances(i));
    
    for c = 1 : 64
        probs(i, c) = gauss_distribution(c - 1, mu, sigma);
    end
    
    % Normalize the probabilities.
    total = sum(probs(i, :));
    for c = 1 : 64
        probs(i, c) = probs(i, c) / total;
    end
end

disp('Class probabilities calculated.');

perplex = perplexity(probs, y_data, 0);

disp('Perplexity calculated.');
perplex


%% Using a Neural Network - Setup
disp('5 (Neural Network).');

% Tidy up any previous state.
clear all
close all

% The base of the file locations.
base = '/afs/inf.ed.ac.uk/group/teaching/mlprdata/challengedata/';

% If the user is not on dice, they have to enter the location of the file
% manually.
try
    load(strcat(base, 'imdata.mat'));
catch %#ok<CTCH>
    disp('Unable to find imdata.mat on afs. Please enter location: ');
    location = input('? ', 's');
    load(location);
end

% Convert to double and remove un-needed columns. Renamed to avoid
% conflicts later on if xmtestdata.mat is loaded.
x_data = double(x(:, [end, end - 33, end - 34, end - 35]));
y_data = double(y);

% Free up some memory.
clear 'i' 'x' 'y' 'base' 'location';

disp('Data loaded.');

%% Using a Neural Network - Main

% The number of input, hidden, and output nodes.
nin = 4;
nhidden = 3;
nout = 1;

% Netlab is needed for mlp, etc.
path(path, 'netlab');

% Split the data into 4 pieces.
split_size = length(x_data) / 4;
x_split = mat2cell(x_data, repmat(split_size, 4, 1), size(x_data, 2));
y_split = mat2cell(y_data, repmat(split_size, 4, 1), size(y_data, 2));

% Perform the 4-fold cross validation linear regression.
y_cell = cell(4);
vs = zeros(4, 1);
for i = 1 : 4
    x_validation = x_split;
    x_validation(i) = [];
    x_validation = cell2mat(x_validation);
    
    y_validation = y_split;
    y_validation(i) = [];
    y_validation = cell2mat(y_validation);
    
    net = mlp(nin, nhidden, nout, 'linear');
    
    options = zeros(1,18);
    
    [net, ~] = netopt(net, options, x_validation, y_validation, 'scg');
    mlp_output = mlpfwd(net, x_validation);
    vs(i) = mean((y_validation - mlp_output).^2);
    
    x_test = cell2mat(x_split(i));
    y_cell{i} = mlpfwd(net, x_test);
end

ys = cell2mat(y_cell);

% Replicate the vs to match them row-by-row to the ys.
variances = reshape(repmat(vs', split_size, 1), [], 1);

% To calculate the output probabilities, create a gaussian centered around
% each output y value, with a variance drawn from the respective row in 
% variances.
probs = zeros(length(ys), 64);
for i = 1 : length(ys)
    mu = ys(i);
    sigma = variances(i);
    
    for c = 1 : 64
        probs(i, c) = gauss_distribution(c - 1, mu, sigma);
    end
end

disp('Class probabilities calculated.');

perplex = perplexity(probs, y_data, 0);

disp('Perplexity calculated.');
perplex