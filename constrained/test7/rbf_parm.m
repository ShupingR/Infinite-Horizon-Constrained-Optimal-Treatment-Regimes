% hyperparm selection for centers and scales of radial basis functions
% The simplest approach, randomly select a number of training examples,
% as RBF centers. This method has the advantage of being very fast, but
% the network will likely require an excessive number of centers. Once 
% the center positions have been selected, the spread parameters ?k can
% be estimated, for instance, from the average distance between neighboring
% centers

function [ cent, dist ] = rbf_parm(sample)
    
    % retrieve the states for all the samples
    sample_state = vertcat(sample.state); 
    s = size( sample_state, 1 );

    cent = zeros(4,1);
    dist = zeros(4, s);
    cent(1) = quantile(sample_state, 0.20);
    cent(2) = quantile(sample_state, 0.40);
    cent(3) = quantile(sample_state, 0.60);
    cent(4) = quantile(sample_state, 0.80);
    [ ~, dist(1,:) ] =  knnsearch(sample_state , cent(1), 'k', s, 'distance','euclidean');
    [ ~, dist(2,:) ] =  knnsearch(sample_state , cent(2), 'k', s, 'distance','euclidean');
    [ ~, dist(3,:) ] =  knnsearch(sample_state , cent(3), 'k', s, 'distance','euclidean');
    [ ~, dist(4,:) ] =  knnsearch(sample_state , cent(4), 'k', s, 'distance','euclidean');
    dist = mean( dist, 2 );

end
