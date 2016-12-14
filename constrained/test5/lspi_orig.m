function weight = lspi_orig( train_sample, k, gamma,  which_reward)
    % train_sample : training sample
    % k : dimemsion of features 
    % gamma : discount factor
    % which_r is used to pick which rewards to use
    maxiter = 1000;
    epsilon = 0.0001;

    %%% Initialize policy iter 
    iter = 0;
    distance = inf;
    weight = ones(5*(1+k),1);

    while ( (iter < maxiter) && (distance > epsilon) )  
        %%% Initialize variables
        iter = iter + 1;
        disp('*********************************************************');
        disp( ['LSPI iter : ', num2str(iter)] );
    
        A = zeros( 5*(1+k), 5*(1+k));
        b = zeros( 5*(1+k), 1);
    
        pre_weight = weight;    
    
        mytime = cputime;

        %%% loop through the samples to construct A and b
        i = 1;
        for s = train_sample
            %each (s, a, r, s')              
            r = s.negative_reward * ( which_reward == -1 )+ ...
                 s.positive_reward * ( which_reward ==  1 );
            [~, next_act] = max ( sum(vec2mat( s.next_phi .* weight', k+1), 2) );      
            next_pick = zeros( 5, k+1);
            next_pick( next_act, : ) =  1;
            next_phi = vec2mat( vec2mat( s.next_phi, k+1 ) .* next_pick, 1)';
            
            %%% Update the matrices A and b
            A = A + s.phi'* (s.phi - gamma * next_phi); 

            b = b + s.phi' * r;

            i = i + 1;
        end
        disp('matrix A');
        disp(A);
        disp('vector b');
        disp(b);
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

        solve_time = cputime - mytime;
        disp(['CPU time to solve Aw=b : ' num2str(solve_time)]);
        %%%----- End: : LSTDq, update weight -------------------------------%%%

        difference = weight - pre_weight;
       % LMAXnorm = norm(difference,inf);
       % L2norm = norm(difference);
        distance = norm(difference); %L2norm;

        %%% Print some information
        %disp(['Norms -> Lmax : ', num2str(LMAXnorm), ...
        %      'L2 : ', num2str(L2norm)]);

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
