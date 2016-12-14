function weight = policy_eval_sm( tau, train_sample, k, gamma,  which_reward, delta)
    % - tau: indexing the policy
    % - train_sample: sample data for training, data should be normalized and scaled, 
    %   including the basis feature,
    % - K:  total dimension of radial basis function feature vector, total
    % is |A|*(k+1), k is for each action with out one, A has 4 levels
    % - which_reward : which qfunction to evaluate, posq or negq
    % - gamma : discount factor
   
    %%% Initialize policy iter 
    B = (1/delta) * eye( 5* (k+1) );
    b = zeros( 5* (k+1) , 1);
    mytime = cputime;

        
    %    policy_act = policy( tau,  next_phi); 
     %   next_phi_pol = [1, this_sample.next_phi(policy_act, :)];

    for s = train_sample
        % each (s, a, r, s')
        %%% Compute the action according to the policy under evaluation
        % and the corresponding basis at the next state
        % max phi'tau 
        % [~, next_act] = max( sum( vec2mat( s.next_phi  , k+1) .* vec2mat( tau, k+1), 2) );
        % softmax
        % hand pick threshold for discretize actions max 144 min -144
        % action 0, 0.25, 0.5, 0.75, 1c        
        % next_act = ( s.next_phi(1:4) * tau < -86 ) * 1 + ...
                       ( s.next_phi(1:4) * tau > -86 && s.next_phi(1:4) * tau < -28 ) * 2 + ...
                       ( s.next_phi(1:4) * tau > -28 && s.next_phi(1:4) * tau < 28 ) * 3 + ...
                       ( s.next_phi(1:4) * tau > 28 && s.next_phi(1:4) * tau < 86 ) * 4 + ...
                       ( s.next_phi(1:4) * tau > 86 ) * 5;
        % display(sum( vec2mat( s.next_phi  , k+1) .* vec2mat( tau, k+1), 2) );
        pick = zeros( 5, k+1); 
        pick( next_act, : ) = 1;
        nextphi_hat = vec2mat( vec2mat( s.next_phi, k+1 ) .* pick, 1 )';
       % disp(nextphi_hat);
        numB = B * s.phi' * ( s.phi - gamma * nextphi_hat ) * B ; 
        
        denB = 1 + ( s.phi - gamma * nextphi_hat ) * B * s.phi' ;
        B = B - numB / denB;
        r = s.negative_reward * ( which_reward == -1 )+ ...
             s.positive_reward * ( which_reward ==  1 );
        b = b + s.phi'*r;
    end    
    
    weight = B * b;

    solve_time = cputime - mytime;
    disp(['CPU time to solve w=Bb : ' num2str(solve_time)]);

end