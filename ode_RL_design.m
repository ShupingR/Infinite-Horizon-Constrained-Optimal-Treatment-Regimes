%---------------------------- 
% replicate the simulated dataset using ode modeling in reinforcement 
% learning clinical design. 
% ref: Reinforcement learning design for cancer clinical trials
% Stat Med. 2009 November 20; 28(26): 3294?3315. doi:10.1002/sim.3720.
% Oct 17
%-----------------------------


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
N = 50;

% t = 0, 1,..., T-1, T = 6 => t = 1, 2, ..., 6, 7 months
T = 7;

% Dt, action variable, the chemotherapy agent dose level
% Wt, state variable, measures the negative part of wellness (toxicity)
% Mt, state variable, denotes the tumor size at time t
% Create Matrix of W M D
W = nan(N, T);
M = nan(N, T);

% The initial values W0 and M0 for each patient are generated iid
% from uniform (0, 2). D0 iid ~ uniform (0.5, 1), Dt iid ~ uniform(0,1)
% In general, you can generate N random numbers in the interval (a,b) 
% with the formula r = a + (b-a).*rand(N,1).
% t = 0
W(:,1) = 0 + (2 - 0) .* rand(N,1);
M(:,1) = 0 + (2 - 0) .* rand(N,1); 
D(:,1) = 0.5 + (1 - 0.5) .* rand(N,1);
D(:,2:T-1) = 1 + (1 - 0) .* rand(N,T-2);
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
R_1 = nan(N, T-1);
R_2 = nan(N, T-1);
R_3 = nan(N, T-1);
for t = 1:T-1
    R_1(:, t) = - 60 .* (F(:, t+1) == 1);
    R_2(:, t) =   5 .* ( W(:, t+1) - W(:,t) <= -0.5) ...
                - 5 .* ( W(:, t+1) - W(:,t) > -0.5);
    R_3(:, t) =  15 .* ( M(:, t+1) == 0 ) ...
                + 5 .* ((M(:, t+1) - M(:,t) <= -0.5) & (M(:, t+1) ~= 0))...
                - 5 .* (M(:, t+1) - M(:,t) > 0.5);
end

%% Q-learning replication
R = R_1 + R_2 + R_3;

figure
for i = 1:N
    hold on
    plot(1:6, R(i,:), 'color', 'y');
end

%%% Initialize variables
k = 8;
A = zeros(k, k);
b = zeros(k, 1);
discount = 0.8;
mytime = cputime;
med_m = median(reshape(M, 1, N*T));
med_w = median(reshape(W, 1, N*T));

 %%% Loop through the samples
for t = 1:T
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
        r = R(i,t);
        %%% Compute the basis for the current state and action
        %%% phi = feature(new_policy.basis, samples(i).state, samples(i).action);
        phi = feature( m, w, d, f, med_w, med_m );

        % retrieve next state in the sample quadruplet
        nxt_m = M(i,t+1);
        nxt_w = W(i,t+1);
        nxt_f = F(i,t+1);
           
        %%% Compute the action according to the policy under evaluation
        % and the corresponding basis at the next state 
        nxt_d = policy(nxt_m, nxt_w, nxt_f); 
        nxt_phi = feature(nxt_m, nxt_w, nxt_d, nxt_f, med_w, med_m);
     
        %%% Update the matrices A and b
        A = A + phi * (phi - discount * nxt_phi)'; 
        b = b + phi * r;
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
    w = A\b;
  else
    disp(['WARNING: A is lower rank!!! Should be ' num2str(k)]);
    w = pinv(A)*b;
  end
  
  solve_time = cputime - mytime;
  disp(['CPU time to solve Aw=b : ' num2str(solve_time)]);

  
return
  

