function [ probability ] = gauss_distribution(x, m, s)
%GAUSS_DISTRIBUTION Computes the value of a gaussian at a point x.
%   The gaussian is defined by mean m and standard deviation s.

exponent = -.5 * ((x - m)/s) .^ 2;
normalizer = (s * sqrt(2*pi));

probability = exp(exponent) ./ normalizer;

end