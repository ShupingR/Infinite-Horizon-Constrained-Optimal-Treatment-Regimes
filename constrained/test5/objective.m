function obj = objective( tau, train_sample, k, gamma, which_reward, delta, sign ) %, T )
    % positive reward 1 objective
    % negative reward -1 constraint
    %tau = tau / norm(tau);
    disp('--------------------------------------------------------');
    disp('current tau :');
    disp(tau');
    weight = policy_eval_sm( tau, train_sample, k, gamma,  which_reward, delta);
    
    val_pol = 0;
    for s = train_sample
        next_act = ( s.next_phi(1:4) * tau < -86 ) * 1 + ...
                       ( s.next_phi(1:4) * tau > -86 && s.next_phi(1:4) * tau < -28 ) * 2 + ...
                       ( s.next_phi(1:4) * tau > -28 && s.next_phi(1:4) * tau < 28 ) * 3 + ...
                       ( s.next_phi(1:4) * tau > 28 && s.next_phi(1:4) * tau < 86 ) * 4 + ...
                       ( s.next_phi(1:4) * tau > 86 ) * 5;
        % display(sum( vec2mat( s.next_phi  , k+1) .* vec2mat( tau, k+1), 2) );
        pick = zeros( 5, k+1); 
        pick( next_act, : ) = 1;
        phi = vec2mat( vec2mat( s.phi, k+1 ) .* pick, 1 )';
       % disp(nextphi_hat);        pol_s = policy(tau, [ones(6,1), s.phi]);
        qval_s_pi_s = phi * weight;
        val_pol = val_pol + qval_s_pi_s;
    end
    
    obj = sign *  val_pol / s(1).sample_size;
    
end
     