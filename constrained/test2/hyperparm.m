% hyperparm selection for centers and scales of radial basis functions
% The simplest approach, randomly select a number of training examples,
% as RBF centers. This method has the advantage of being very fast, but
% the network will likely require an excessive number of centers. Once 
% the center positions have been selected, the spread parameters ?k can
% be estimated, for instance, from the average distance between neighboring
% centers

function center_sig_mat = hyperparm(sample, K, nb, seed)

    % random pick centers and calculate average distance in P neighbours
    % TK : number of features TK = 2 R * 6 A * 10 K
    % P : number of neighbours used to calculate sigma, the average
    %     distance
    % generate center and sigma based on action groups
    rng(seed,'twister');
    
    % retrieve the states for all the samples
    sample_state = horzcat(vertcat(sample.wellness), ...
                           vertcat(sample.tumorsize), ...
                           vertcat(sample.action));
    
    sample_state_a1 = sample_state( sample_state(:,3) == 0.0 , : );
    sample_state_a2 = sample_state( sample_state(:,3) == 0.2 , : );
    sample_state_a3 = sample_state( sample_state(:,3) == 0.4 , : );
    sample_state_a4 = sample_state( sample_state(:,3) == 0.6 , : );
    sample_state_a5 = sample_state( sample_state(:,3) == 0.8 , : );
    sample_state_a6 = sample_state( sample_state(:,3) == 1.0 , : );
    
    % random generating indices for centers pick out from sample data

    center_index_a1 = randi( size(sample_state_a1, 1), K, 1 );
    center_index_a2 = randi( size(sample_state_a2, 1), K, 1 );
    center_index_a3 = randi( size(sample_state_a3, 1), K, 1 );
    center_index_a4 = randi( size(sample_state_a4, 1), K, 1 );
    center_index_a5 = randi( size(sample_state_a5, 1), K, 1 );
    center_index_a6 = randi( size(sample_state_a6, 1), K, 1 );
    
    % row represent action, feature is a row vector 6*K, 1 for wellness,
    % 2 for tumorsize
    center_sig_mat = nan(6, K, 3);
    center_sig_mat(:,:, 3) = ones(6, K);
    center_sig_mat(:, :, 1) = vertcat( ...
                          sample_state_a1(center_index_a1, 1)', ... 
                          sample_state_a2(center_index_a2, 1)', ...
                          sample_state_a3(center_index_a3, 1)', ...
                          sample_state_a4(center_index_a4, 1)', ...
                          sample_state_a5(center_index_a5, 1)', ...
                          sample_state_a6(center_index_a6, 1)' ); 

   
    center_sig_mat(:, :, 2) = vertcat( ...
                          sample_state_a1(center_index_a1, 2)', ... 
                          sample_state_a2(center_index_a2, 2)', ...
                          sample_state_a3(center_index_a3, 2)', ...
                          sample_state_a4(center_index_a4, 2)', ...
                          sample_state_a5(center_index_a5, 2)', ...
                          sample_state_a6(center_index_a6, 2)' ); 
                      
    % calculate average neighbouring distance for each center
    
    % 3 for sigma
    for j = 1:K
        center_a1 = [center_sig_mat(1, j, 1), center_sig_mat(1, j, 2)];
        [ ~, d_a1 ] = knnsearch(sample_state_a1(:,1:2) , center_a1, ...
                      'k', nb, 'distance','euclidean');
        center_sig_mat(1, j, 3) = mean(d_a1);
        
        center_a2 = [center_sig_mat(2, j, 1), center_sig_mat(2, j, 2)];
        [ ~, d_a2 ] = knnsearch(sample_state_a1(:,1:2) , center_a2, ...
                      'k', nb, 'distance','euclidean');
        center_sig_mat(2, j, 3) = mean(d_a2);
        
        center_a3 = [center_sig_mat(3, j, 1), center_sig_mat(3, j, 2)];
        [ ~, d_a3 ] = knnsearch(sample_state_a3(:,1:2) , center_a3, ...
                      'k', nb, 'distance','euclidean');
        center_sig_mat(3, j, 3) = mean(d_a3);

        center_a4 = [center_sig_mat(4, j, 1), center_sig_mat(4, j, 2)];
        [ ~, d_a4 ] = knnsearch(sample_state_a4(:,1:2) , center_a4, ...
                      'k', nb, 'distance','euclidean');
        center_sig_mat(4, j, 3) = mean(d_a4);
        
        center_a5 = [center_sig_mat(5, j, 1), center_sig_mat(5, j, 2)];
        [ ~, d_a5 ] = knnsearch(sample_state_a5(:,1:2) , center_a5, ...
                      'k', nb, 'distance','euclidean');
        center_sig_mat(5, j, 3) = mean(d_a5);
        
        center_a6 = [center_sig_mat(6, j, 1), center_sig_mat(6, j, 2)];
        [ ~, d_a6 ] = knnsearch(sample_state_a6(:,1:2) , center_a6, ...
                      'k', nb, 'distance','euclidean');
        center_sig_mat(6, j, 3) = mean(d_a6);
    end
    
end
