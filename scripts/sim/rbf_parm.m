%---------------------------------------------------------------------------%
% hyperparm selection for centers and scales of radial basis functions
% The simplest approach, use 20% 40% 60% 80% percentile Once
% the center positions have been selected, the spread parameters'k can
% be estimated, for instance, from the average distance between neighboring
% centers
%---------------------------------------------------------------------------%
function [ cent, dist ] = rbf_parm(state_mat, K)
    
    % retrieve the states for all the samples
    sample_state = state_mat(:); 
    s = size( sample_state, 1 );

    cent = zeros(K,1);
    dist = zeros(K, s);
    for k = 1:K
        cent(k) = prctile(sample_state, k/(K+1)*100);
        [ ~, dist(k,:) ] =  knnsearch(sample_state , cent(k), 'k', s, ...
                               'distance','euclidean');
    end

   dist = mean( dist, 2 );

end
