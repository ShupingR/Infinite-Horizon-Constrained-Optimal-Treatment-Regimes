function weight = policy_eval( tau, sample, all_phi, center, sigma2,...
                               which_reward, K )
    % which_r is used to pick which rewards to use
    % batch samples, and K total feature dimension
    % dimensions of basis function vector
    maxiter = 500;
    epsilon = 0.0001;
    discount = 0.8;

    %%% Initialize policy iter 
    iter = 0;
    distance = inf;
    weight = ones(K,1);
    
    while ( (iter < maxiter) && (distance > epsilon) )  
        %%% Initialize variables
        iter = iter + 1;
        disp('*********************************************************');
        disp( ['LSq iter : ', num2str(iter)] );
    
        A = zeros(K, K);
        b = zeros(K, 1);
    
        pre_weight = weight;    
    
        mytime = cputime;

        %%% loop through the samples to construct A and b
        i = 1;
        for this_sample = sample
            % the observation here is (s, a, r, s')
            % s = [ M(i,t), W(i,t), F(i,t) ];
            % a = d = D(i,t)
            % r = R(i,t);
            % s' = [ M(i,t+1), W(i,t+1), F(i,t+1)];
            % retrieve current state in the sample quadruplet


            %%% Compute the basis for the current state and action
            %%% phi = feature(new_policy.basis, samples(i).state, samples(i).action);
           % disp('** sample **');
           % display(i);
            phi = feature(this_sample, center, sigma2, this_sample.action,...
                          which_reward, K);
            %%% reward get from choose action d at current state s
            r = this_sample.negative_reward * ( which_reward == -1 )+ ...
                this_sample.positive_reward * ( which_reward ==  1 );
            
            %%% Compute the action according to the policy under evaluation
            % and the corresponding basis at the next state
            nxt_act = policy( all_phi(i,:), tau ); 
            nxt_phi = feature( this_sample, center, sigma2, nxt_act, ...
                      which_reward, K);
    
            %%% Update the matrices A and b
            A = A + phi * (phi - discount * nxt_phi)'; 
            b = b + phi * r;    
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
            disp(['WARNING: A is lower rank!!! Should be ' num2str(k)]);
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
