%-------------------------------------------------------------------- 
% funtion for data generation 
% replicate the simulated dataset using ode modeling in ref
% ref: Reinforcement learning design for cancer clinical trials
% Stat Med. 2009 November 20; 28(26): 3294?3315. doi:10.1002/sim.3720.
% Oct 24, 2016
%--------------------------------------------------------------------

% input : N sample size
%         T total stage number: 1, ... , T in code (= 0, ..., T-1 in paper)
% output: class sample : (s, a, r, s') where s = (w, m, f)

% S_1, ..., S_(T-1), S_T with S_t = (M_t, W_t); T = 7 based on paper, 6
% month duration, 7 decision point; last decision point only has state, no
% action, no reward
% A_1, ..., A_(T-1)
% R_1, ..., R_(T-1)

function test_sample = test(N, T, tau, seed, center_sig_mat, K)
    rng(seed,'twister');
    %% ode modeling for tumor size and patient wellness (negative/toxicity)
    % params set
    a1 = 0.1;
    a2 = 0.15;
    b1 = 1.2;
    b2 = 1.2;
    d1 = 0.5;
    d2 = 0.5;
    
    W = nan(N, T); % wellness
    M = nan(N, T); % tumorsize
    D = nan(N, T-1); % action/dose
    F = nan(N, T); % death indicator
    % The initial values W1 (W0) and M1 (M0) for each patient are generated 
    % iid from uniform (0, 2). 
    W(:,1) = 0 + (2 - 0) .* rand(N,1);
    M(:,1) = 0 + (2 - 0) .* rand(N,1); 
    F(:,1) = zeros(N,1); % assume no one is dead at the beginning point t =1

    % patient survival status create a matrix for p of death
    p_mat = nan(N, T); % at the end of each interval/action t=1,2,3,4,5,6
    % p(1) = 0;
    p_mat(:,1) = 0;
    % parms in log hazard modeling
    mu0 = -8.5;
    mu1 = 1;
    mu2 = 1;
        
    for i = 1: N
        for t = 1: T-1 % Last T only has state21
        % Dt, action variable, the chemotherapy agent dose level 
        % Wt, state variable, measures the negative part of wellness (toxicity)
        % Mt, state variable, denotes the tumor size 
        % all at decision time t
        % Create Matrix of W M D
        % D1 (D0) iid ~ uniform (0.5, 1), Dt iid ~ uniform(0,1).
        % In general, you can generate N random numbers in the interval (a,b) 
        % with the formula r = a + (b-a).*rand(N,1).
            phi = i_feature_construction(W(i,t), M(i,t), F(i,t), center_sig_mat, K);
            tempD = policy(tau, [ones(6,1), phi]);
            D(i,t) = ( tempD == 1 ) * 0.0 + ( tempD == 2 ) * 0.2 + ...
                       ( tempD == 3 ) * 0.4 + ( tempD == 4 ) * 0.6  + ...
                       ( tempD == 5 ) * 0.8 + ( tempD == 6 ) * 1.0;
            % Wellness M, Tumorsize W, D dose/action
            M_t1 = [M(i,t), M(i,1)];
            W_t1 = [W(i,t), W(i,1)];
            
            dW_t = a1 * max(M_t1, [], 2) + b1 * (D(i,t) - d1);
            dM_t = ( a2 * max(W_t1, [], 2) - b2 * ( D(i,t) - d2) ) .* ( M(i,t) > 0 );
            
            W(i,t+1) = max( W(i,t) + dW_t, 0 );
            M(i,t+1) = max( M(i,t) + dM_t, 0 ); 
            
            % F dead or not
            if (t > 1)
                p_mat(i, t) = 1 - exp(-exp(mu0 + mu1 * W(i ,t) + mu2 * M(i,t)));
                F(i, t) = (rand(1, 1) < p_mat(i, t) | F(i , t-1) ==1);
            end

        end
    end
    for i = 1:N
        p_mat(i, T) = 1 - exp(-exp(mu0 + mu1 * W(i ,T) + mu2 * M(i,T)));
        F(i, T) = (rand(1, 1) < p_mat(i, T) | F(i, T-1) ==1);
    end
    % display(p_mat);
    %% calculate the rewards
    neg_r = nan(N, T-1);
    pos_r = nan(N, T-1);
    for t = 1:T-1
        neg_r(:, t) = 60 * (F(:, t+1) == 1) + 5 * (W(:, t+1) - W(:,t) >= -0.5 & F(:, t+1) ~= 1);
        pos_r(:, t) = 0 * (F(:, t+1) == 1) + ...
                         5 * (M(:, t+1) - M(:,t) <= -0.5 & M(:, t+1) ~= 0 & F(:, t+1) ~= 1 ) + ...
                         15 * (M(:, t+1) == 0 & F(:, t+1) ~= 1);
    end
    
    max_M = max(max(M));
    min_M = min(min(M));
    upp_M = max_M + abs(max_M - min_M)/10;
    low_M = min_M - abs(max_M - min_M)/10;
    max_W = max(max(W));
    min_W = min(min(W));
    upp_W = max_W + abs(max_W - min_W)/10;
    low_W = min_W - abs(max_W - min_W)/10;
    test_sample = SampleClass;
    q = 1;
    for i = 1:N
        for t = 1:T-1
            % state
            test_sample(q).wellness = W(i,t);
            test_sample(q).tumorsize = M(i,t);
            test_sample(q).death = F(i,t);
            test_sample(q).phi =  ...
                i_feature_construction(W(i,t), M(i,t), F(i,t), center_sig_mat, K);
            % state range
            test_sample(q).upp_wellness = upp_W;
            test_sample(q).low_wellness = low_W;
            test_sample(q).upp_tumorsize = upp_M;
            test_sample(q).low_tumorsize = low_M;
            % action : dose
            test_sample(q).action = D(i,t);
            % nxt_state
            test_sample(q).next_wellness = W(i,t+1);
            test_sample(q).next_tumorsize = M(i,t+1);
            test_sample(q).next_death = F(i,t+1);
            test_sample(q).next_phi = ...
                i_feature_construction(W(i,t+1), M(i,t+1), F(i,t+1), center_sig_mat, K);
            % reward
            test_sample(q).negative_reward = neg_r(i,t);
            test_sample(q).positive_reward = pos_r(i, t);
            test_sample(q).sample_size = N * (T-1);
            q = q+1;
        end
    end
end