% objective function  for positive rewards based on tumor size
function [c, ceq] = constraint_function( tau, sample_neg, kappa)  %, T )
    % positive reward 1 objective
    % negative reward -1 constraint
    %tau = tau / norm(tau);    
    algorithm = 1;  
    maxiteration = 100;
    epsilon = 0.001;
    initial_policy_weight = ones(5*5 , 1);
    weight = lspi(algorithm, maxiteration, epsilon, sample_neg, initial_policy_weight, tau);
    
    val_pol = 0;
    for each_sample = sample_neg
        state = each_sample.state;
        % phi = basis_rbf(state, 1) + basis_rbf(state, 2) + basis_rbf(state, 3) + ...
        %        basis_rbf(state, 4) + basis_rbf(state, 5);
        [ action, actionphi ]= policy_function_constraint(tau, state);
        % qval_s_pol_s = max(sum( vec2mat(phi .* weight, 5), 2));
        val_pol = val_pol + sum(actionphi .* weight( (action -1)*5 + 1 : action*5));
    end
   
    c =  val_pol / length(sample_neg) - kappa;
    ceq = [];
    
end