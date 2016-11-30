function weight = policy_eval(tau, sample, K, which_reward)
    % which_r is used to pick which rewards to use
    % batch samples, and K total feature dimension
    % dimensions of basis function vector
    maxiter = 100;
    epsilon = 0.0001;
    discount = 0.8;

    %%% Initialize policy iter 
    iter = 0;
    distance = inf;
    weight = ones(1+K,1);
    
    while ( (iter < maxiter) && (distance > epsilon) )  
        %%% Initialize variables
        iter = iter + 1;
        disp('*********************************************************');
        disp( ['LSq iter : ', num2str(iter)] );
    
        A = zeros( 1+K, 1+K);
        b = zeros( 1+K, 1);
    
        pre_weight = weight;    
    
        mytime = cputime;

        %%% loop through the samples to construct A and b
        i = 1;
        for this_sample = sample
            % the observation here is (s, a, r, s')
            %%% Compute the basis for the current state and action
            %%% phi = feature(new_policy.basis, samples(i).state, samples(i).action);
            % disp('** sample **');
            % display(i);
          %  display('**** iteration for calculate weights');
          %  display(i);
            action = this_sample.action;
            act_row = ( action == 0.0 ) * 1 + ( action == 0.2 ) * 2 + ...
                      ( action == 0.4 ) * 3 + ( action == 0.6 ) * 4 + ...
                      ( action == 0.8 ) * 5 + ( action == 1.0 ) * 6;
            phi = [1, this_sample.phi(act_row, :)];
                      
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

        phi_time = cputime - mytime;
        disp(['CPU time to form A and b : ' num2str(phi_time)]);
        mytime = cputime;

        %%% Solve the system to find w
        rankA = rank(A);
    
        rank_time = cputime - mytime;
        disp(['CPU time to find the rank of A : ' num2str(rank_time)]);
        mytime = cputime;

        disp(['Rank of matrix A : ' num2str(rankA)]);
        
        if rankA==min(size(A,1), size(A,2))
            
            disp('A is a full rank matrix!!!');
            weight = A\b;

        else
            
            disp('WARNING: A is lower rank!!! Should be ');
            weight = pinv(A)*b;

        end

        solve_time = cputime - mytime;
        disp(['CPU time to solve Aw=b : ' num2str(solve_time)]);
        %%%----- End: : LSTDq, update weight -------------------------------%%%

        difference = weight - pre_weight;
        LMAXnorm = norm(difference,inf);
        L2norm = norm(difference);
        distance = L2norm;

        %%% Print some information
        disp(['Norms -> Lmax : ', num2str(LMAXnorm), ...
              'L2 : ', num2str(L2norm)]);

    end

    %%% Display some info
    disp('*********************************************************');
    if (distance > epsilon) 
        disp(['LSPI finished in ' num2str(iter) ...
              ' iters WITHOUT CONVERGENCE to a fixed point']);
    else
        disp(['LSPI converged in ' num2str(iter) ' iters']);
    end
end
