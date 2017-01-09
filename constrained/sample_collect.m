% generate sample with neg reward and positive reward together
% normalized and scaled W and M

function sample = sample_collect(N, T, seed)
    % sample size N
    % stage number T
    % k feature dimension for each action
    rng(seed,'twister');
    %% ode modeling for tumor size and patient wellness (negative/toxicity)
    % params set
    a1 = 0.1;
    a2 = 0.15;
    b1 = 1.2;
    b2 = 1.2;
    d1 = 0.5;
    d2 = 0.5;

    % Dt, action variable, the chemotherapy agent dose level 
    % Wt, state variable, measures the negative part of wellness (toxicity)
    % Mt, state variable, denotes the tumor size 
    % all at decision time t
    % Create Matrix of W M D
    W = nan(N, T);
    M = nan(N, T);
    D = nan(N, T-1);
    
    % The initial values W1 (W0) and M1 (M0) for each patient are generated 
    % iid from uniform (0, 2). 
    W(:,1) = 0 + (2 - 0) .* rand(N,1);
    M(:,1) = 0 + (2 - 0) .* rand(N,1); 
    
    %%% D1 (D0) iid ~ uniform (0.5, 1), Dt iid ~ uniform(0,1).
    %%% In general, you can generate N random numbers in the interval (a,b) 
    %%% with the formula r = a + (b-a).*rand(N,1).
    %%% D(:,1) = 0.5 + (1 - 0.5) .* rand(N,1);
    %%% D(:,2:T-1) = 0 + (1 - 0) .* rand(N,T-2);
    

    % D1 = [0.4, 0.6, 0.8, 1.0]';
    % D2 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]';
    
    D1 = [ 0.25, 0.50, 0.75, 1.00 ]';
    D2 = [ 0.00, 0.25, 0.50, 0.75, 1.00 ]';
 
    % D(:,1) = random ( 0.4, 0.6, 0.8, 1.0)
    % D(:,2:T-1) = random (0, 0.2, 0.4, 0.6, 0.8, 1.0)
    D(:, 1) = D1(randi(size(D1,1),N,1),:);
    
    for t = 2: T-1
        D(:, t) = D2(randi(size(D2,1),N,1), :);
    end
    
    for t = 1:T-1
        M_t1 = [M(:,t), M(:,1)];
        W_t1 = [W(:,t), W(:,1)];
        dW_t = a1 * max(M_t1, [], 2) + b1 * (D(:,t) - d1);
        dM_t = ( a2 * max(W_t1, [], 2) - b2 * ( D(:,t) - d2) ) ...
                 .* ( M(:,t) > 0 );
        W(:,t+1) = max( W(:,t) + dW_t, 0 );
        M(:,t+1) = max( M(:,t) + dM_t, 0 );    
    end

    %% patient survival status
    % create a matrix for p
    p_mat = nan(N, T); % at the end of each interval/action t=1,2,3,4,5,6
    % p(1) = 0;
    p_mat(:,1) = 0;
    % parms in log hazard modeling
    mu0 = -8.5;
    mu1 = 1;
    mu2 = 1;
    % create a matrix F for death indicator
    F = nan(N, T);
    % assume no one is dead at the beginning point t =1
    F(:, 1) = 0;
    for t = 2:T
        p_mat(:, t) = 1 - exp(-exp(mu0 + mu1 * W(: ,t) + mu2 * M(:,t)));
        F(:, t) = (rand(N, 1) < p_mat(:, t) | F(: , t-1) ==1);
    end
    % display(p_mat);
    %% calculate the rewards
    pos_r = nan(N, T-1);
    neg_r = nan(N, T-1);
    
    for t = 1:T-1
        pos_r(:, t) = 0 * ( F(:, t+1) == 1) + ...
                         5 * ( M(:, t+1) - M(:,t) <= -0.5 & M(:, t+1) ~= 0 & F(:, t+1) ~= 1 ) + ...
                         15 * ( M(:, t+1) == 0 & F(:, t+1) ~= 1);
        neg_r(:, t) = 60 * (F(:, t+1) == 1) + 5 * ( W(:, t+1) - W(:,t) >= -0.5 & F(:, t+1) ~= 1 );
    end
 
    % standardize input
    disp( mean2(W) );
    disp( std2(W) );
    disp( mean2(M) );
    disp( std2(M) );
    W = ( W - mean2(W) ) / std2(W);
    M = ( M - mean2(M) ) / std2(M) ;

    
    field1 = 'state_pos';  value1 = nan;
    field2 = 'state_neg';  value2 = nan;
    field3 = 'absorb'; value3 = nan;
    field4 = 'action';  value4 = nan;
    field5 = 'reward_pos';  value5 = nan;
    field6 = 'reward_neg';  value6 = nan;
    field7 = 'nextstate_pos';  value7 = nan;
    field8 = 'nextstate_neg';  value8 = nan;
    sample= struct(field1,value1,field2,value2,field3,value3,field4,value4, ...
                         field5,value5, field6,value6, field7,value7, field8,value8);
   
    q = 1;
    for i = 1:N
        for t = 1:T-1
            % all the positive reward samples
            sample(q).state_pos= M(i,t);
            sample(q).state_neg = W(i,t);
            sample(q).absorb = F(i,t);
            sample(q).action =  (D(i,t) == 0) * 1 +  (D(i,t) == 0.25) * 2 + ...
                                       (D(i,t) == 0.5) * 3 + (D(i,t) == 0.75) * 4 + ...
                                       (D(i,t) == 1) * 5;
            sample(q).reward_pos = pos_r(i, t);
            sample(q).reward_neg = neg_r(i, t);
            sample(q).nextstate_pos = M(i,t+1);
            sample(q).nextstate_neg = W(i,t+1);            
            q = q+1;
        end
    end
end