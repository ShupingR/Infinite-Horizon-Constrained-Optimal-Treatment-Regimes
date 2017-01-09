function [action, actionphi_pos, actionphi_neg] = policy_function_two(tau, state_pos, state_neg)

%%%%%%%%%%%%%%%%%%%%
% deterministic polynomial policy function
%%%%%%%%%%%%%%%%%%%%
    in_pol = [1, state_pos, state_pos^2, state_neg, state_neg^2, state_pos*state_neg]  * tau;
    action = (in_pol < -1.5)*1 +  (in_pol > -1.5 && in_pol < -0.5)*2 + ...
            (in_pol >= -0.5 && in_pol < 0.5)*3 + (in_pol >= 0.5 && in_pol < 1.5)*4 + ...
            (in_pol >= 1.5)*5;
    actionphi_pos = basis_rbf(state_pos, action);
    actionphi_neg = basis_rbf(state_neg, action);
end

