% objective function  for positive rewards based on tumor size
function state_val_act = state_valfun( tau, pos_weights, neg_weights, states )  %, T )
    state_val_act = nan(length(states), 3);
    for i = 1: length(states)
        state_pos = states(i, 1);
        state_neg = states(i, 2);
        % phi = basis_rbf(state, 1) + basis_rbf(state, 2) + basis_rbf(state, 3) + ...
        %        basis_rbf(state, 4) + basis_rbf(state, 5);
        [ action, actionphi_pos, actionphi_neg]= policy_function_two(tau, state_pos, state_neg);
        % qval_s_pol_s = max(sum( vec2mat(phi .* weight, 5), 2));
        state_val_act(i, 1) = sum( actionphi_pos .* pos_weights );
        state_val_act(i, 2) = sum( actionphi_neg .* neg_weights);
        state_val_act(i, 3) = action;
    end
end