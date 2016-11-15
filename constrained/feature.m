
%-------------------------------------
% feature construction for Q functions
% input is a single sample
%-------------------------------------
function phi = feature ( sample, center, sigma2, action, which_reward, K )
    % size(center) = ( |phi| * |A| * 2 ), K = | phi | ); 
    if ( sample.death == 1 )
        % absorb state
        phi = zeros(K, 1);
    else
        % pick out the corresponding subset of the features for the action
        which_action = [ (action == 0.0) * ones(1,K), ...
                         (action == 0.2) * ones(1,K), ...
                         (action == 0.4) * ones(1,K), ...
                         (action == 0.6) * ones(1,K), ...
                         (action == 0.8) * ones(1,K), ...
                         (action == 1.0) * ones(1,K) ];
        pick = [ which_action .* (which_reward == 1),...
                 which_action .* (which_reward == -1)] ;
        sample_x = repmat([sample.wellness, sample.tumorsize], K, 1) - ...
                   center(center(:,1) .* pick' ~= zeros(60, 1),:);
        inside = sum(sample_x .* sample_x, 2)  ./ 2;
        phi = exp(-1 * inside);   
    end
    
end