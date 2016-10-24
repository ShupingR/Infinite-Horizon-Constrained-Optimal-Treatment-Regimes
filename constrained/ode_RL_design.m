%-------------------------------------------------------------------- 
% replicate the simulated dataset using ode modeling in reinforcement 
% learning clinical design. 
% ref: Reinforcement learning design for cancer clinical trials
% Stat Med. 2009 November 20; 28(26): 3294?3315. doi:10.1002/sim.3720.
% Oct 17
%--------------------------------------------------------------------
rng(222,'twister');

%-----------------
% data generation

%% ode modeling for tumor size and patient wellness (negative/toxicity)
% params set
a1 = 0.1;
a2 = 0.15;
b1 = 1.2;
b2 = 1.2;
d1 = 0.5;
d2 = 0.5;

% N = 1000 patients
N = 1000;

% t = 0, 1,..., T-1, T = 6 => t = 1, 2, ..., 6, 7 months
T = 7;

% Dt, action variable, the chemotherapy agent dose level
% Wt, state variable, measures the negative part of wellness (toxicity)
% Mt, state variable, denotes the tumor size at time t
% Create Matrix of W M D
W = nan(N, T);
M = nan(N, T);
D = nan(N, T-1);
% The initial values W0 and M0 for each patient are generated iid
% from uniform (0, 2). D0 iid ~ uniform (0.5, 1), Dt iid ~ uniform(0,1)
% In general, you can generate N random numbers in the interval (a,b) 
% with the formula r = a + (b-a).*rand(N,1).
% t = 0
W(:,1) = 0 + (2 - 0) .* rand(N,1);
M(:,1) = 0 + (2 - 0) .* rand(N,1); 
D(:,1) = 0.5 + (1 - 0.5) .* rand(N,1);
D(:,2:T-1) = 0 + (1 - 0) .* rand(N,T-2);
%D = randi([1,3],N, T-1);
%D = 0.25*D;
for t = 1:T-1
    M_t1 = [M(:,t), M(:,1)];
    W_t1 = [W(:,t), W(:,1)];
    dW_t = a1 .*  max(M_t1, [], 2) + b1 .* (D(:,t) - d1);
    dM_t = ( a2 .* max(W_t1, [], 2) - b2 .* (D(:,t) - d2) ).* ( M(:,t) > 0 );
    W(:, t+1) = W(:, t) + dW_t;
    M(:, t+1) = M(:, t) + dM_t;
end

for i = 1:N
     plot(1:7, M(i,:), 'color', 'g');
     hold on
     plot(1:7, W(i,:), 'color', 'b');
 end
% 
figure
 for i = 1:N
     hold on
     plot(1:6, D(i,:), 'color', 'r');
 end


%% patient survival status
% create a matrix for p
p_mat = nan(N, T); % at the end of each interval/action t=1,2,3,4,5,6,7
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
    p_mat(:, t) = 1 - exp(-exp(mu0 + mu1 .* W(: ,t) + mu2 .* M(:,t)));
    F(:, t) = (rand(N, 1) < p_mat(:, t) | F(: , t-1) ==1);
end

%% calculate the rewards
R_neg = nan(N, T-1);
R_pos= nan(N, T-1);

for t = 1:T-1
    R_neg(:, t) = 60 * (F(:, t+1) == 1) ...
                  + 5 * (W(:, t+1) - W(:,t) >= -0.5);
    R_pos(:, t) = 5 * (M(:, t+1) - M(:,t) <= -0.5) ...
                  + 15 * (M(:, t+1) == 0);
end


 figure
 for i = 1:N
     hold on
     plot(1:6, R_neg(i,:), 'color', 'y');
 end
 
 
 
 figure
 for i = 1:N
     hold on
     plot(1:6, R_pos(i,:), 'color', 'y');
 end

% dimensions of basis function vector
k = 9;
maxiter = 500;
epsilon = 0.005;
discount = 0.8;

%%% Initialize policy iter 
iter = 0;
distance = inf;
weights_pos = ones(k,1);
while ( (iter < maxiter) && (distance > epsilon) )  
    %%% Initialize variables
    iter = iter + 1;
    disp('*********************************************************');
    disp( ['LSPI iter : ', num2str(iter)] );
    
    %%%----- Begin: LSTDq, update weights ----------------------------%%%
    A = zeros(k, k);
    b = zeros(k, 1);
    
    pre_weights_pos = weights_pos;    
    
    mytime = cputime;
    mean_m = mean(reshape(M, 1, N*T));
    mean_w = mean(reshape(W, 1, N*T));

    %%% loop through the samples
    for t = 1:T-1
        for i=1:N
            % the observation here is (s, a, r, s')
            % s = [ M(i,t), W(i,t), F(i,t) ];
            % a = d = D(i,t)
            % r = R(i,t);
            % s' = [ M(i,t+1), W(i,t+1), F(i,t+1)];
            % retrieve current state in the sample quadruplet
            m = M(i,t);
            w = W(i,t);
            f = F(i,t);
            d = D(i,t); % action: dose level
            r_pos = R_pos(i,t);

            %%% Compute the basis for the current state and action
            %%% phi = feature(new_policy.basis, samples(i).state, samples(i).action);
            phi = feature( m, w, d, f, mean_m, mean_w,k);

            % retrieve next state in the sample quadruplet
            nxt_m = M(i,t+1);
            nxt_w = W(i,t+1);
            nxt_f = F(i,t+1);

            %%% Compute the action according to the policy under evaluation
            % and the corresponding basis at the next state 
            nxt_d = policy(nxt_m, nxt_w, nxt_f, mean_m, mean_w, weights_pos, k); 
            nxt_phi = feature(nxt_m, nxt_w, nxt_d, nxt_f, mean_m, mean_w, k);
            
            
            %%% Update the matrices A and b
            A = A + phi * (phi - discount * nxt_phi)'; 
            b_pos = b_pos + phi * r_pos;

        end
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
    if rankA==k
        disp('A is a full rank matrix!!!');
        weights_pos = A\b_pos;

    else
        disp(['WARNING: A is lower rank!!! Should be ' num2str(k)]);
        weights_pos = pinv(A)*b_pos;

    end
    
    solve_time = cputime - mytime;
    disp(['CPU time to solve Aw=b : ' num2str(solve_time)]);
    %%%----- End: : LSTDq, update weight -------------------------------%%%

    difference_pos = weights_pos - pre_weights_pos;
    LMAXnorm = norm(difference_pos,inf);
    L2norm = norm(difference_pos);
    distance_pos = L2norm;
    
    %%% Print some information
    disp(['Norms -> Lmax : ', num2str(LMAXnorm), ...
          'L2 : ', num2str(L2norm)]);
    
end


%%% Initialize policy iter 
iter = 0;
distance = inf;
weights_neg = ones(k,1);
while ( (iter < maxiter) && (distance > epsilon) )  
    %%% Initialize variables
    iter = iter + 1;
    disp('*********************************************************');
    disp( ['LSPI iter : ', num2str(iter)] );
    
    %%%----- Begin: LSTDq, update weights ----------------------------%%%
    A = zeros(k, k);
    b = zeros(k, 1);
    
    pre_weights_neg = weights_neg;    
    
    mytime = cputime;
    mean_m = mean(reshape(M, 1, N*T));
    mean_w = mean(reshape(W, 1, N*T));

    %%% loop through the samples
    for t = 1:T-1
        for i=1:N
            % the observation here is (s, a, r, s')
            % s = [ M(i,t), W(i,t), F(i,t) ];
            % a = d = D(i,t)
            % r = R(i,t);
            % s' = [ M(i,t+1), W(i,t+1), F(i,t+1)];
            % retrieve current state in the sample quadruplet
            m = M(i,t);
            w = W(i,t);
            f = F(i,t);
            d = D(i,t); % action: dose level
            r_neg = R_neg(i,t);

            %%% Compute the basis for the current state and action
            %%% phi = feature(new_policy.basis, samples(i).state, samples(i).action);
            phi = feature( m, w, d, f, mean_m, mean_w,k);

            % retrieve next state in the sample quadruplet
            nxt_m = M(i,t+1);
            nxt_w = W(i,t+1);
            nxt_f = F(i,t+1);

            %%% Compute the action according to the policy under evaluation
            % and the corresponding basis at the next state 
            nxt_d = policy(nxt_m, nxt_w, nxt_f, mean_m, mean_w, weights, k); 
            nxt_phi = feature(nxt_m, nxt_w, nxt_d, nxt_f, mean_m, mean_w, k);
            
            
            %%% Update the matrices A and b
            A = A + phi * (phi - discount * nxt_phi)'; 
            b_neg = b_neg + phi * r_neg;

        end
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
    if rankA==k
        disp('A is a full rank matrix!!!');
        weights_neg = A\b_neg;

    else
        disp(['WARNING: A is lower rank!!! Should be ' num2str(k)]);
        weights_neg = pinv(A)*b_neg;
    end
    
    solve_time = cputime - mytime;
    disp(['CPU time to solve Aw=b : ' num2str(solve_time)]);
    %%%----- End: : LSTDq, update weight -------------------------------%%%

    difference_neg = weights_neg - pre_weights_neg;
    LMAXnorm = norm(difference_neg,inf);
    L2norm = norm(difference_neg);
    distance_pos = L2norm;
    
    %%% Print some information
    disp(['Norms -> Lmax : ', num2str(LMAXnorm), ...
          'L2 : ', num2str(L2norm)]);
    
end

%%% Display some info
disp('*********************************************************');
if (distance_neg > epsilon) 
    disp(['LSPI finished in ' num2str(iter) ...
          ' iters WITHOUT CONVERGENCE to a fixed point']);
else
    disp(['LSPI converged in ' num2str(iter) ' iters']);
end

% %%% est_q vs d plot, mean m , mean w
% mean_m = mean(reshape(M, 1, N*T));
% mean_w = mean(reshape(W, 1, N*T));
% d_list = linspace(0, 1, 30);
% qval_list = nan(1, 30);
% f1 = 0;
% l = 1;
% figure
% for dplot = d_list 
%     qval_list(l) = qfun(mean_m, mean_w, dplot, f1, med_m, med_w, weights, k);
%     l = l+1;
% end
% 
% plot(d_list, qval_list);
% hold on
% %%% est_q vs d plot, quantile 25
% q25_m = prctile(reshape(M,1,N*T), .25);
% q25_w = prctile(reshape(W,1,N*T), .25);
% d_list = linspace(0, 1, 30);
% qval_list = nan(1, 30);
% f1 = 0;
% l = 1;
% for dplot = d_list 
%     qval_list(l) = qfun(q25_m, q25_w, dplot, f1, med_m, med_w, weights, k);
%     l = l+1;
% end
% 
% plot(d_list, qval_list);
% hold on
% %%% est_q vs d plot, quantile 25
% q75_m = prctile(reshape(M,1,N*T), .75);
% q75_w = prctile(reshape(W,1,N*T), .75);
% d_list = linspace(0, 1, 30);
% qval_list = nan(1, 30);
% f1 = 0;
% l = 1;
% for dplot = d_list 
%     qval_list(l) = qfun(q75_m, q75_w, dplot, f1, med_m, med_w, weights, k);
%     l = l+1;
% end
% 
% plot(d_list, qval_list);
