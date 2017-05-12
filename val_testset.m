% Apply a estimated indexing parameter and weights to a new testset 
function [ val_pos, val_neg ] = val_testset(tau, pos_weight, neg_weight, sample ) %, T )
    % positive reward 1 objective
    % negative reward -1 constraint
    % tau policy indices and weights for q functions are estimated from
    % training dataset
    val_pos = 0;
    val_neg = 0;
    for s = sample
        state_pos = s.state_pos;
        state_neg = s.state_pos;
        [ ~, actionphi_pos, actionphi_neg ] = ...
            policy_function_two(tau, state_pos, state_neg);
        qval_pos = actionphi_pos' * pos_weight;
        qval_neg = actionphi_neg' * neg_weight;
        val_pos = val_pos + qval_pos;
        val_neg = val_neg + qval_neg;
    end
    
    val_pos =  val_pos / length(sample);
    val_neg =  val_neg / length(sample);
    
end
     