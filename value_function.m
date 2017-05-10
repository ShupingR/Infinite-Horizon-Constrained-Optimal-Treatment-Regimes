% objective function  for positive rewards based on tumor size
function [ obj, weight ] = value_function(tau, sample, discount, K, L, which_reward, sign)  %, T )
    % input : 
    % tau : policy index
    % sample: sample with defined structure
    % which_reward: 1 positive reward, -1 negative reward
    % K: number of radial basis functions plus 1 for intercept
    % L: number of dosage levels
    % sign: sign = 1, maximize constraint_function; sign = -1, minimize constraint_function  
    % output:
    % obj value
    % weight for q function for corresponding rewards
    weight = lsq(tau, sample, discount, K, L, which_reward);
    
    pol_val = 0;
    if ( which_reward == 1 ) 
        for each_sample = sample
            [ ~, actionphi_pos, ~]= policy_function_deterministic(tau, each_sample);
            pol_val = pol_val + sum(actionphi_pos .* weight);
        end
        obj = sign * pol_val / length(sample);
    elseif( which_reward == -1 )
        for each_sample = sample
            [ ~, ~, actionphi_neg]= policy_function_deterministic(tau, each_sample);
            pol_val = pol_val + sum(actionphi_neg .* weight);
        end
        obj = sign * pol_val / length(sample);
    else
        disp('reward not specified');
    end
    
end
