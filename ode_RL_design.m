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

% t = 0, 1,..., T-1, T = 6 => t = 1, 2, ..., 6 months
T = 6;

% Dt, action variable, the chemotherapy agent dose level
% Wt, state variable, measures the negative part of wellness (toxicity)
% Mt, state variable, denotes the tumor size at time t
% Create Matrix of W M D
W = nan(N, T+1);
M = nan(N, T+1);
D = nan(N, T);

% The initial values W0 and M0 for each patient are generated iid
% from uniform (0, 2). D0 iid ~ uniform (0.5, 1), Dt iid ~ uniform(0,1)
% In general, you can generate N random numbers in the interval (a,b) 
% with the formula r = a + (b-a).*rand(N,1).
% t = 0
W(:,1) = 0 + (2 - 0) .* rand(N,1);
M(:,1) = 0 + (2 - 0) .* rand(N,1); 
D(:,1) = 0.5 + (1 - 0.5) .* rand(N,1);
D(:,2:T) = 1 + (1 - 0) .* rand(N,T-1);

for t = 1:T
    M_t1 = [M(:,t), M(:,1)];
    W_t1 = [W(:,t), W(:,1)];
    dW_t = a1 .*  max(M_t1, [], 2) + b1 .* (D(:,t) - d1);
    dM_t = ( a2 .* max(W_t1, [], 2) - b2 .* (D(:,t) - d2) ).* ( M(:,t) > 0 );
    W(:, t+1) = W(:, t) + dW_t;
    M(:, t+1) = M(:, t) + dM_t;
end

% for i = 1:N
%     plot(1:7, M(i,:), 'color', 'g');
%     hold on
%     plot(1:7, W(i,:), 'color', 'b');
% end

% figure
% for i = 1:N
%     hold on
%     plot(1:6, D(i,:), 'color', 'r');
% end


%% patient survival status
% create a matrix for p
p_mat = nan(N, T+1); % at the end of each interval/action t=1, 2,3,4,5,6,7
% p(1) = 0;
p_mat(:,1) = 0;
% parms in log hazard modeling
mu0 = -9;
mu1 = 1;
mu2 = 1;
% create a matrix F for death indicator
F = nan(N, T+1);
% assume no one is dead at the beginning point t =1
F(:, 1) = 0;
for t = 2:T+1
    p_mat(:, t) = 1 - exp(-exp(mu0 + mu1 .* W(: ,t) + mu2 .* M(:,t)));
    F(:, t) = (rand(N, 1) < p_mat(:, t) | F(: , t-1) ==1);
end
% F(:, 2:T+1) = (rand(N, T) < p_mat(:, 2:T+1)); 


%% Q-learning replication
