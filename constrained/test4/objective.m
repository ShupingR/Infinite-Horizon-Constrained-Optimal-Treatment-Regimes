function obj = objective(tau, sample, K, which_reward, sign ) %, T )
    % positive reward 1 objective
    % negative reward -1 constraint
    tau = tau / norm(tau);
    weight = policy_eval(tau, sample, K, which_reward);
    
    val_pol = 0;
    for each_sample = sample
        pol_s = policy(tau, [ones(6,1), each_sample.phi]);
        qval_s_pol_s = [1, each_sample.phi(pol_s,:)] * weight;
        val_pol = val_pol + qval_s_pol_s;
    end
    
    obj = sign *  val_pol / sample(1).sample_size;
    
end
     