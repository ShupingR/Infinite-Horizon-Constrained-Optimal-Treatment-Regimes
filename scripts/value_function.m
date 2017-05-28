% value function for either reward
function [ val, weight ] = value_function(tau, sample, discount, K, L, which_reward, sign)  %, T )
    % input : 
    % tau : policy index
    % sample: sample with defined structure
    % which_reward: 1 positive reward, -1 negative reward
    % sign: sign = -1, to maximize function, final value need to flip sign; 
    %        sign = 1, to minimize function  
    % output:
    % val: policy value overall V * sign
    % weight for q function for corresponding rewards
    weight = lsq(tau, sample, discount, K, L, which_reward);
    
    pol_val = 0;
    if ( which_reward == 1 ) 
        for each_sample = sample
            [ ~, actionphi_pos, ~]= policy_function_deterministic(tau, each_sample);
            pol_val = pol_val + sum(actionphi_pos .* weight);
        end
        val = sign * pol_val / length(sample);
    elseif( which_reward == -1 )
        for each_sample = sample
            [ ~, ~, actionphi_neg]= policy_function_deterministic(tau, each_sample);
            pol_val = pol_val + sum(actionphi_neg .* weight);
        end
        val = sign * pol_val / length(sample);
    else
        disp('reward not specified');
    end
    
end
