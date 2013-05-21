function perplexity = perplexity( x, y, log_probs )
%PERPLEXITY Calculates the perplexity of a set of inputs.
%   x is an n-by-c matrix, where each row is the set of probabilities
%   for the classes 0, ..., c-1. (That is, class 0 is in column 1, class
%   1 in column 2, ...). 
%   
%   y is an n-by-1 matrix of the correct classes for each row of x.
%
%   log_probs is an indicator variable. If set, the input x values are
%   already negative log probabilities.

% Default to non-log probabilities.
if (nargin < 3) 
    log_probs = 0;
end

% Adjust the classes in y to index the x columns correctly.
y_adj = y + 1;

% Select the correct probabilities.
probs = x(sub2ind(size(x), (1:numel(y_adj))', y_adj));

% log P(x_n | M).
if (~log_probs)
    probs = log(probs);
end

% 1/N_t * sum_n (log P(x_n | M)).
mean_prob = mean(probs);

% exp(-1/N_t * sum_n (log P(x_n | M))).
if (~log_probs)
    perplexity = exp(-mean_prob);
else
    perplexity = exp(mean_prob);
end

end

