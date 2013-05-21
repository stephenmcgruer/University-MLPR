%% Setup
disp('Setup.');

% Tidy up any previous state.
clear all
close all

% The base of the file locations on afs.
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

% Free up some memory.
clear 'i' 'base' 'location';

x = double(x);
y = double(y);

disp('Data loaded.');

%% PCA
disp('PCA.');

% If the PCA has been run before, the file pca.mat contains the
% mean, covariance, reconstructed x, and error.
if exist('pca.mat', 'file')
    disp('Existing pca.mat file found, loading data...');
    load pca.mat
    
    % Still need to calculate the eigenvectors.
    [V, D] = eigs(S);
    eigenvectors = V(:, 1:3);
    disp('Done.')
else
    disp('No existing file found. Doing PCA from scratch...');
    
    m = mean(x);
    S = cov(x);
    disp(' Mean and covariance calculated.');

    % Get the first three eigenvectors.
    [V, D] = eigs(S);
    eigenvectors = V(:, 1:3);
    disp('Eigenvectors calculation.')

    % Reconstruct x from the first 3 eigenvalues.
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
    disp('x_r constructed.');
    
    % Calculated the mean squared errors.
    difference = (x - x_r);
    e = (sum(difference .^ 2, 2)) ./ size(difference, 2);
    
    disp('Errors calculated. Saving data...');
    
    % Save the PCA.
    save ('pca.mat', 'm', 'S', 'x_r', 'e');
    
    disp('Data saved.');
end

%% 2a, 2b
disp('2a, 2b.');

% Display the mean image patch.
mean_image = reshape([m zeros(1,18)], 35, 30)';
imagesc(mean_image);
title('Mean Image Patch.');
xlabel('X-Position (Pixels)');
ylabel('Y-Position (Pixels)');
pause

% Diplay the three principal components.
e1_image = reshape([eigenvectors(:, 1)' zeros(1,18)], 35, 30)';
e2_image = reshape([eigenvectors(:, 2)' zeros(1,18)], 35, 30)';
e3_image = reshape([eigenvectors(:, 3)' zeros(1,18)], 35, 30)';

imagesc(e1_image);
title('First Principal Vector Image Patch.');
xlabel('X-Position (Pixels)');
ylabel('Y-Position (Pixels)');
pause

imagesc(e2_image);
title('Second Principal Vector Image Patch.');
xlabel('X-Position (Pixels)');
ylabel('Y-Position (Pixels)');
pause

imagesc(e3_image);
title('Third Principal Vector Image Patch.');
xlabel('X-Position (Pixels)');
ylabel('Y-Position (Pixels)');
pause

% Display the best represented image patch.
[~, I] = min(e);
best_image = reshape([x(I, :) zeros(1,18)], 35, 30)';
best_image_r = reshape([x_r(I, :) zeros(1,18)], 35, 30)';
imagesc(best_image);
title('Best image.');
xlabel('X-Position (Pixels)');
ylabel('Y-Position (Pixels)');
pause

imagesc(best_image_r);
title('Best image (reconstructed).');
xlabel('X-Position (Pixels)');
ylabel('Y-Position (Pixels)');
pause

% Display the worst represented image patch.
[~, I] = max(e);
worst_image = reshape([x(I, :) zeros(1,18)], 35, 30)';
worst_image_r = reshape([x_r(I, :) zeros(1,18)], 35, 30)';
imagesc(worst_image);
title('Worst image.');
xlabel('X-Position (Pixels)');
ylabel('Y-Position (Pixels)');
pause

imagesc(worst_image_r);
title('Worst image (reconstructed).');
xlabel('X-Position (Pixels)');
ylabel('Y-Position (Pixels)');
pause

%% 2c, 2d
disp('2c, 2d.');

% Plot a histogram of the targets.
[n xout] = hist(y, 64);
bar(xout, n);
axis([0 63 0 max(n)*1.1]);
title('Histogram of the target values.');
xlabel('Value of y');
ylabel('Number of Occurences');
pause

% Compute y(i) - x(i, end), and display a histogram.
difference = y - x(:, end);
[n xout] = hist(difference, 64);
bar(xout, n);
axis([min(xout) max(xout) 0 max(n)*1.1]);
title('Histogram of the values of y - x(:, end).');
xlabel('Value of y - x(:, end)');
ylabel('Number of Occurences');
pause

% Plot a histogram of x(:, end).
[n xout] = hist(x(:, end), 64);
bar(xout, n);
axis([0 63 0 max(n)*1.1]);
title('Histogram of the values of x(:, end).');
xlabel('Value of x(:, end)');
ylabel('Number of Occurences');
pause