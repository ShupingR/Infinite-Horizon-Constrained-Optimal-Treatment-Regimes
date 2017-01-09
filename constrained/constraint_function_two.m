% objective function  for positive rewards based on tumor size
function [c, ceq] = constraint_function_two( tau, sample, kappa)  %, T )
    % positive reward 1 objective
    % negative reward -1 constraint
    %tau = tau / norm(tau);    
    algorithm = 1;  
    maxiteration = 100;
    epsilon = 0.001;
    initial_policy_weight = ones(5*5 , 1);
    weight = lspi_two_neg(algorithm, maxiteration, epsilon, sample, initial_policy_weight, tau);
    
    val_pol = 0;
    for each_sample = sample
        state_pos = each_sample.state_pos;
        state_neg = each_sample.state_neg;
        % phi = basis_rbf(state, 1) + basis_rbf(state, 2) + basis_rbf(state, 3) + ...
        %        basis_rbf(state, 4) + basis_rbf(state, 5);
        [ ~, ~, actionphi_neg ]= policy_function_two(tau, state_pos, state_neg);
        % qval_s_pol_s = max(sum( vec2mat(phi .* weight, 5), 2));
        val_pol = val_pol + sum(actionphi_neg .* weight);
    end
   
    c =  val_pol / length(sample) - kappa;
    ceq = [];
    
end