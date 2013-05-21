%% Setup
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
clear 'i' 'x' 'y' 'base' 'location';

% Split the data into 4 sets for cross validation.
split_length = length(x_data) / 4;
x_split = mat2cell(x_data, repmat(split_length, 4, 1), size(x_data, 2));
y_split = mat2cell(y_data, repmat(split_length, 4, 1), size(y_data, 2));

disp('Data loaded.');

%% 3bi
disp('3bi.');

% As far as I am able to tell, placing a Dirichlet distribution across
% the hyperparameters with alpha = 1 simply reduces to applying Laplacian
% ('plus one') smoothing to both the class conditional and prior
% probabilities.

% Having read the forums and seen Amos' description and other people's
% attempts, I'm now fairly sure my initial derivation was wrong. So I
% apologise for the following code, since it's probably therefore complete
% rubbish. I just don't get Dirichlet...

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

%% 3bii
disp('3bii.');

probs = cell2mat(prob_out);

% Some interesting cases to plot.

% A standard case where Naive Bayes places all of it's weight on the
% correct answer.
h = bar(0:63, probs(833,:));

% Not sure how this works, but it seems like in this case the bars are
% at 2, 7, 12, ... So to convert we multiply the value by 5 and add 2.
h_child = get(h,'Children');
fvcd = get(h_child,'FaceVertexCData');
fvcd((y_data(833) * 5) + 2) = 255;
set(h_child, 'FaceVertexCData', fvcd)

axis([-1 64 0 max(probs(833, :))*1.1]);
xlabel('Y');
ylabel('Negative Log Probability');
pause;

% A standard case where Naive Bayes places all of it's weight on the
% wrong answer.
h = bar(0:63, probs(4837,:));

% Not sure how this works, but it seems like in this case the bars are
% at 2, 7, 12, ... So to convert we multiply the value by 5 and add 2.
h_child = get(h,'Children');
fvcd = get(h_child,'FaceVertexCData');
fvcd((y_data(4837) * 5) + 2) = 255;
set(h_child, 'FaceVertexCData', fvcd)

axis([-1 64 0 max(probs(4837, :))*1.1]);
xlabel('Y');
ylabel('Negative Log Probability');
pause;

% A case where Naive Bayes has no idea what to do, so comes up with
% a fairly uniform probability distribution.
h = bar(0:63, probs(82948,:));

% Not sure how this works, but it seems like in this case the bars are
% at 2, 7, 12, ... So to convert we multiply the value by 5 and add 2.
h_child = get(h,'Children');
fvcd = get(h_child,'FaceVertexCData');
fvcd((y_data(82948) * 5) + 2) = 255;
set(h_child, 'FaceVertexCData', fvcd)

axis([-1 64 0 max(probs(82948, :))*1.1]);
xlabel('Y');
ylabel('Negative Log Probability');
pause;

%% 3biv
disp('3biv.');

% The base of the file locations.
base = '/afs/inf.ed.ac.uk/group/teaching/mlprdata/challengedata/';

% If the user is not on dice, they have to enter the location of the file
% manually.
try
    load(strcat(base, 'imtestdata.mat'));
catch %#ok<CTCH>
    disp('Unable to find imtestdata.mat on afs. Please enter location: ');
    location = input('? ', 's');
    load(location);
end

% Convert to double and remove un-needed columns.
x_test = double(x(:, [end, end - 34, end - 35]));

% Free up some memory.
clear 'i' 'x' 'base' 'location';

disp('Data loaded.');

[outputs, priors]  = train_naivebayes(x_data, y_data);

disp('Classifier trained.');

kaggle_probs = test_naivebayes(x_test, outputs, priors);

% Offset to be based at 0.
kaggle_mins = min(kaggle_probs, [], 2);
kaggle_probs = kaggle_probs - ...
    repmat(kaggle_mins, 1, size(kaggle_probs, 2));

disp('Probabilities computed.');

csvwrite('naivebayes.csv', kaggle_probs);

disp('CSV file written.');