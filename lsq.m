function weights = lsq(tau, sample, discount, K, L, which_reward)
    % input: 
    % samples: samples generated 
    % tau: policy index
    % 
    % which reward: postive reward = 1 or negative reward = -1
    % output: 
    % weights for q function approximation under a fixed policy index
    
    %%% If no samples, return
    if isempty(sample)
        disp('Warning: Empty sample set');
        return
    end
 
    %%% specify LSQ : choose postive reward or negative reward
    if (which_reward == 1)
       [ weights, ~, ~] = lsq_pos(tau, sample, discount, K, L);
    elseif (which_reward == -1)
       [ weights, ~, ~] = lsq_neg(tau, sample, discount, K, L);
    end 
end
