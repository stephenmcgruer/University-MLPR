%% Setup.
disp('Setup.');

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
x_data = double(x(:, [end, end - 34, end - 35]));
y_data = double(y);

% Free up some memory.
clear 'i' 'x' 'y';

disp('Data loaded.');

%% 4a
disp('4a - Linear Regression on the original x.');

% Compute the output y values and variances using linear regression.
[ys, variances] = n_fold_linear_regression(4, x_data, y_data);

disp('Linear regression values calculated.');

% To calculate the output probabilities, create a gaussian centered around
% each output y value, with a variance drawn from the respective row in 
% variances.
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

perplex_4a = perplexity(probs, y_data, 0);

disp('Perplexity calculated.');
perplex_4a


%% 4b - Linear Regression on the reconstructed data
disp('4b - Linear Regression on the reconstructed data');

% Re-load imdata to get access to the full x-data.
try
    load(strcat(base, 'imdata.mat'));
catch %#ok<CTCH>
    load(location);
end

% Free up some memory.
clear 'i', 'y';

x = double(x);

% Now do PCA over the x_data. If the PCA has been run before, the file 
% pca_linear.mat contains the reconstructed x.
disp('Beginning PCA.');
if exist('pca_linear.mat', 'file')
    disp('Existing pca_linear.mat file found, loading data...');
    
    load pca_linear.mat
    
    disp('Done.');
else
    disp('No existing file found. Doing PCA from scratch...');

    m = mean(x);
    S = cov(x);
    disp(' Mean and covariance calculated.');

    % Calculate the eigenvectors.
    [V, D] = eigs(S, 10);
    eigenvectors = V(:, 1:10);
    disp('Eigenvectors calculated.');
    
    % Reconstruct x from the first 10 eigenvectors.
    % Loop to save on memory - the run-time is slower but it shouldnt
    % run out of memory.
    x_r = zeros(size(x,1), size(x,2));
    tic
    disp('Calculating the reconstructed x...');
    for i = 1 : size(x, 1)
        % Inform the user every 5000 calculations.
        if (mod(i, 5000) == 0)
            disp(strcat('Loop ', num2str(i)))
            toc
            tic
        end
        e_x_m = repmat((eigenvectors' * (x(i,:) - m)')', ...
            size(eigenvectors, 1), 1);
        x_r(i, :) = m + sum(e_x_m .* eigenvectors, 2)';
    end
    toc
    
    % Remove un-needed columns.
    x_r = x_r(:, [end, end - 34, end - 35]);
    disp('x_r constructed. Saving data...');
    
    % Save the reconstructed x_r.
    save('pca_linear.mat', 'x_r');
    
    disp('Data saved.');
end

% Now run linear regression.

% Compute the output y values and variances using linear regression.
[ys, variances] = n_fold_linear_regression(4, x_r, y_data);

disp('Linear regression values calculated.');

% To calculate the output probabilities, create a gaussian centered around
% each output y value, with a variance drawn from the respective row in 
% variances.
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

perplex_4b = perplexity(probs, y_data, 0);

disp('Perplexity calculated.');
perplex_4b