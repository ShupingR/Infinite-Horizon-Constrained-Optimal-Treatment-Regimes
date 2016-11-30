% hyperparm selection for centers and scales of radial basis functions
% The simplest approach, randomly select a number of training examples,
% as RBF centers. This method has the advantage of being very fast, but
% the network will likely require an excessive number of centers. Once 
% the center positions have been selected, the spread parameters ?k can
% be estimated, for instance, from the average distance between neighboring
% centers

function [ center, sigma2 ] = hyperparm(sample, TK, P, seed)
    % K : number of features K = 2 R * 6 A * | phi_R_A |
    % P : number of neighbours used to calculate sigma2
    rng(seed,'twister');
    % initialize center and sigma
    center = nan(TK, 2);
    sigma2 = nan(TK, 1);
    % retrieve the states for all the samples
    sample_state = horzcat(vertcat(sample.wellness),vertcat(sample.tumorsize));
    % random generating indices for centers pick out from sample data
    center_indices = randi(sample(1).sample_size, 1, TK);
    
    k = 1;
    for center_index = center_indices
        % calculate the average distance between enighthboring centers
        this_center = [ sample(center_index).wellness,...
                        sample(center_index).tumorsize ];
        center(k, :) = this_center; 
        [ ~, D ] = knnsearch(sample_state , this_center, ...
                             'k', P, 'distance','euclidean');
        sigma2(k) = (mean(D))^2;
        k = k+1;
    end
    
end
