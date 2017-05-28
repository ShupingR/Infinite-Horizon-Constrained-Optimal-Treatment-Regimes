% Apply a estimated indexing parameter and weights to a new testset 
function [ state_val_pos, state_val_neg ] = state_val(tau, pos_weight, neg_weight, sample ) %, T )
    % positive reward 1 objective
    % negative reward -1 constraint
    % tau policy indices and weights for q functions are estimated from
    % training dataset
    state_val_pos = 0;
    state_val_neg = 0;
    for s = sample
        state_pos = s.state_pos;
        state_neg = s.state_neg;
        [ ~, actionphi_pos, actionphi_neg ] = ...
            policy_function_two(tau, state_pos, state_neg);
        qstate_val_pos = actionphi_pos' * pos_weight;
        qstate_val_neg = actionphi_neg' * neg_weight;
        state_val_pos = state_val_pos + qstate_val_pos;
        state_val_neg = state_val_neg + qstate_val_neg;
    end
    
    state_val_pos =  state_val_pos / length(sample);
    state_val_neg =  state_val_neg / length(sample);
    
end
     