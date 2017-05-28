%------------------------------------------------------------------------------%
% constraint function for negative rewards based on negative wellness/toxicity %
%------------------------------------------------------------------------------%
function [c, ceq] = constraint(tau, sample, discount, K, L, nu)
     % input : 
    % tau : policy index
    % sample: sample with defined structure
    % which_reward: 1 positive reward, 0 negative reward
    % K: number of radial basis functions plus 1 for intercept
    % L: number of dosage levels
    % sign: sign = 1, maximize constraint_function; sign = -1, minimize constraint_function
    % nu: constraint value
    % output:
    % constraint value
%     % weight for q function negative reward   
    which_reward = -1; % negative reward
    weight = lsq(tau, sample, discount, K, L, which_reward);
    
    val_pol = 0;
    for each_sample = sample
        [ ~, ~, actionphi_neg ]= policy_function_deterministic(tau, each_sample);
        val_pol = val_pol + sum(actionphi_neg .* weight);
    end
   
    c = val_pol / length(sample) - nu;
    ceq = [ ];
end
