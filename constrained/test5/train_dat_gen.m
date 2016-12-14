%-------------------------------------------------------------------- 
% funtion for data generation 
% replicate the simulated dataset using ode modeling in ref
% ref: Reinforcement learning design for cancer clinical trials
% Stat Med. 2009 November 20; 28(26): 3294?3315. doi:10.1002/sim.3720.
% Oct 24, 2016
%--------------------------------------------------------------------

% input : N sample size, T total stage number: 1, ... , T in code (= 0, ..., T-1 in paper)
% output: class sample : (s, a, r, s') where s = (w, m, f)

% S_1, ..., S_(T-1), S_T with S_t = (M_t, W_t); T = 7 based on paper, 6
% month duration, 7 decision point; last decision point only has state, no
% action, no reward
% A_1, ..., A_(T-1)
% R_1, ..., R_(T-1)

function sample = train_dat_gen(N, T, k, seed, show)
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
    neg_r = nan(N, T-1);
    pos_r = nan(N, T-1);

    for t = 1:T-1
        neg_r(:, t) = 60 * (F(:, t+1) == 1) + 5 * ( W(:, t+1) - W(:,t) >= -0.5 & F(:, t+1) ~= 1 );
        pos_r(:, t) = 0 * ( F(:, t+1) == 1) + ...
                         5 * ( M(:, t+1) - M(:,t) <= -0.5 & M(:, t+1) ~= 0 & F(:, t+1) ~= 1 ) + ...
                         15 * ( M(:, t+1) == 0 & F(:, t+1) ~= 1);
    end
 
    % standardize input
    W = ( W - mean2(W) ) / std2(W);
    M = ( M - mean2(M) ) / std2(M) ;
    
    % center and sigma
    max_M = max(max(M));
    min_M = min(min(M));
    center_M = linspace( min_M, max_M, k )';
    [ ~, sig_M] = ...
        knnsearch( reshape(M, [], 1), center_M, 'k', N*T, 'distance', 'euclidean');
    sig_M = mean( sig_M, 2);

    max_W = max(max(W));
    min_W = min(min(W));
    center_W = linspace( min_W, max_W, k )';
    [ ~, sig_W] = ...
        knnsearch(  reshape(W, [], 1), center_W, 'k', N*T, 'distance', 'euclidean');
    sig_W = mean( sig_W, 2);
    state_center = horzcat( center_M, center_W ); % long matrix
    state_sig = horzcat( sig_M, sig_W ); % long matrix
    
    sample = SampleClass;
    q = 1;
    for i = 1:N
        for t = 1:T-1
            % state
            sample(q).wellness = W(i,t);
            sample(q).tumorsize = M(i,t);
            sample(q).death = F(i,t);
            state = [W(i,t) , F(i,t)];
            state_mat = repmat ( state, k, 1); % long matrix
            phi = exp( sum( ( ( state_mat - state_center ) ./ state_sig ).^2, 2) ); % long matrix
            phi = repmat( [1, phi'], 5, 1 );
            pick = zeros( 5, k+1);
            sample(q).action = D(i,t);
            act = sample(q).action;
            act_row = 1 * ( act == 0 ) + 2 * ( act == 0.25 ) + 3 * ( act == 0.5 ) + ...
                           4 * ( act == 0.75 ) + 5 * ( act == 1);
            pick( act_row, : ) =  1;
            phi = vec2mat( vec2mat( phi, k+1 ) .* pick, 5*(k+1));
            
            sample(q).phi = ( F(i,t) == 1 ) * zeros(1, ( k+1 )*5 ) + ...
                                  ( F(i,t) ~= 1 ) * phi;


            % nxt_state
            sample(q).next_wellness = W(i,t+1);
            sample(q).next_tumorsize = M(i,t+1);
            sample(q).next_death = F(i,t+1);
            next_state = [W(i,t+1) , F(i,t+1)];
            next_state_mat = repmat ( next_state, k, 1);
            next_phi = exp( sum( ( ( next_state_mat - state_center ) ./ state_sig ).^2, 2) );
            next_phi = repmat( vertcat(1, next_phi), 5, 1 )';
            sample(q).next_phi = ( F(i,t+1) == 1 ) * zeros(1, ( k+1 )*5 ) + ...
                                         ( F(i,t+1) ~= 1 ) * next_phi ;
            % reward
            sample(q).negative_reward = neg_r(i,t);
            sample(q).positive_reward = pos_r(i, t);
            sample(q).sample_size = N * (T-1);
            q = q+1;
        end
    end
    if (show == 1)
        display('*********** sample *************');
        for q = 1:N*(T-1)
            display (sample(q));
        end
        
        figure
        title('Tumor size (green)/ Toxicity (blue) vs. Time');
        for i = 1:N
            hold on
            plot(1:T, M(i,:), 'color', 'g');
            hold on
            plot(1:T, W(i,:), 'color', 'b');
        end
        
        figure
        title('Dose (red) vs. Time');
        for i = 1:N
            hold on
            plot(1:T-1, D(i,:), 'color', 'r');
        end

    
        figure
        title('Positive (cyan) / negative (magetta) reward vs. Time');
        for i = 1:N
            hold on
            plot(1:T-1, pos_r(i,:), 'color', 'c');
            hold on
            plot(1:T-1, neg_r(i,:), 'color', 'm');
        end
    end
end