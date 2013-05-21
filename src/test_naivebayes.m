function [ probabilities ] = test_naivebayes( x, parameters, priors )
%input_row_NAIVEBAYES Computes the negative log posteriors for a set of input features using multinomial naive bayes.
%   Assumes that there are 64 classes.
%
%   x is a n-by-f matrix where each row is an input that a posterior should
%   be computed for.

probabilities = zeros(length(x), 64);
for j = 1 : length(x)
    input_row = x(j, :);

    % Calculate the posterior for each class, using a log-likelihood
    % method.
    for c = 1 : 64
        prob = log(priors(c));
        for f = 1 : size(parameters, 3)
            prob = prob + log(parameters(c, input_row(f) + 1, f));
        end
        probabilities(j, c) = -(prob);
    end
end

end

