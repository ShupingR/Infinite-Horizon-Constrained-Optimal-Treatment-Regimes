function weight = lspi_sm( train_sample, k, gamma,  which_reward)
    % train_sample : training sample
    % k : dimemsion of features 
    % gamma : discount factor
    % which_r is used to pick which rewards to use
   
    %%% Initialize policy iter 
    maxiter = 500;
    epsilon = 0.0001;
    delta = 0.01;
    %%% Initialize policy iter 
    iter = 0;
    distance = inf;
    weight = zeros(5*(1+k),1);
    
    B = (1/delta) * eye( 5* (k+1) );
    b = zeros( 5* (k+1) , 1);
    mytime = cputime;

    while ( (iter < maxiter) && (distance > epsilon) )  
        %%% Initialize variables
        iter = iter + 1;
        disp('*********************************************************');
        disp( ['LSPI iter : ', num2str(iter)] );
        disp(weight');
        pre_weight = weight;  
        %    policy_act = policy( tau,  next_phi); 
        %   next_phi_pol = [1, this_sample.next_phi(policy_act, :)];

        for s = train_sample
            % each (s, a, r, s')
            r = s.negative_reward * ( which_reward == -1 )+ ...
            s.positive_reward * ( which_reward ==  1 );
            pick = zeros( 5, k+1); 
            [~, next_act] = max ( sum(vec2mat( s.next_phi .* weight', k+1), 2) );      
            pick( next_act, : ) = 1;
            nextphi_hat = vec2mat( vec2mat( s.next_phi, k+1 ) .* pick, 1 )';
            % disp(nextphi_hat);
            numB = B * s.phi' * ( s.phi - gamma * nextphi_hat ) * B ; 
        
            denB = 1 + ( s.phi - gamma * nextphi_hat ) * B * s.phi' ;
            B = B - numB / denB;

            b = b + s.phi'*r;
        end    
    
        weight = B * b;

        solve_time = cputime - mytime;
        disp(['CPU time to solve w=Bb : ' num2str(solve_time)]);
      
        difference = weight - pre_weight;
        % LMAXnorm = norm(difference,inf);
        % L2norm = norm(difference);
        distance = norm(difference); %L2norm;
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