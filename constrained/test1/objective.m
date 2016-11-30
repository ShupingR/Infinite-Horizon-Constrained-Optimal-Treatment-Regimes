function obj = objective( sample, all_phi, center, sigma2, tau, K ) %, T )
    % positive reward
    which_reward = 1;
                       
    action_pol = policy ( all_phi, tau );
              
    which_action = [ (action_pol == 0.0) * ones(1,K), ...
                     (action_pol == 0.2) * ones(1,K), ...
                     (action_pol == 0.4) * ones(1,K), ...
                     (action_pol == 0.6) * ones(1,K), ...
                     (action_pol == 0.8) * ones(1,K), ...
                     (action_pol == 1.0) * ones(1,K) ];
                 
    pick = [ which_action .* (which_reward == 1), ...
             which_action .* (which_reward == -1) ] ;
    pick_mat = repmat(pick, size(all_phi,1),1);
    feature_pol = all_phi .* pick_mat; 
    
    weight = policy_eval( tau, sample, all_phi, center, sigma2,...
                          which_reward, K );
                      
    nonzero_feature_pol = nan(sample(1).sample_size, K);
    for i = 1:sample(1).sample_size
      row_feature_pol = feature_pol(i,:);
      nonzero_feature_pol(i,:) = row_feature_pol(row_feature_pol~=0);
    end
    
    obj = -1 * mean( nonzero_feature_pol * weight ); 
end
     