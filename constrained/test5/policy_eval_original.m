function weight = policy_eval_original( tau, train_sample, k, gamma,  which_reward, delta)
    % which_r is used to pick which rewards to use
    % batch samples, and K total feature dimension
    % dimensions of basis function vector

    A = zeros( 4*(1+k), 4*(1+k) );
    b = zeros( 4*(1+k), 1);
    %mytime = cputime;

    %%% loop through the samples to construct A and b
    i = 1;
    for this_sample = train_sample
        % the observation here is (s, a, r, s')
        action = this_sample.action;
        act_row = ( action == 0.0 ) * 1 + ( action == 0.25 ) * 2 + ...
                       ( action == 0.5 ) * 3 + ( action == 0.75 ) * 4 + ...
                       ( action == 1 ) * 5;
        phi = this_sample.phi;

        %%% reward get from choose action d at current state s
        r = this_sample.negative_reward * ( which_reward == -1 )+ ...
             this_sample.positive_reward * ( which_reward ==  1 );

        %%% Compute the action according to the policy under evaluation
        % and the corresponding basis at the next state
        next_phi = [ones(6,1), this_sample.next_phi];
       % policy_act = policy(tau, pre_weight, next_phi); 
        policy_act = policy( tau,  next_phi); 
        %pol_act_row = ( policy_act == 0.0 ) * 1 + ...
        %              ( policy_act == 0.2 ) * 2 + ...
        %              ( policy_act == 0.4 ) * 3 + ...
        %              ( policy_act == 0.6 ) * 4 + ...
        %              ( policy_act == 0.8 ) * 5 + ...
        %              ( policy_act == 1.0 ) * 6;            
        next_phi_pol = [1, this_sample.next_phi(policy_act, :)];

        %%% Update the matrices A and b
        A = A + phi'* (phi - discount * next_phi_pol); 
        b = b + phi' * r;
        i = i + 1;
    end

   % phi_time = cputime - mytime;
   % disp(['CPU time to form A and b : ' num2str(phi_time)]);
   % mytime = cputime;

    %%% Solve the system to find w
    rankA = rank(A);

   % rank_time = cputime - mytime;
   % disp(['CPU time to find the rank of A : ' num2str(rank_time)]);
  %  mytime = cputime;

   % disp(['Rank of matrix A : ' num2str(rankA)]);

    if rankA==min(size(A,1), size(A,2))

       % disp('A is a full rank matrix!!!');
        weight = A\b;

    else

        %disp('WARNING: A is lower rank!!! Should be ');
        weight = pinv(A)*b;

    end

    %solve_time = cputime - mytime;
    %disp(['CPU time to solve Aw=b : ' num2str(solve_time)]);
    %%%----- End: : LSTDq, update weight -------------------------------%%%

end

