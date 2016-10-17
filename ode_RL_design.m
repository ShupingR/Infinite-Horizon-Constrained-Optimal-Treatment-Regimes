%---------------------------- 
% replicate the simulated dataset using ode modeling in reinforcement 
% learning clinical design. 
% ref: Reinforcement learning design for cancer clinical trials
% Stat Med. 2009 November 20; 28(26): 3294?3315. doi:10.1002/sim.3720.
% Oct 17
%-----------------------------


rng(222,'twister');

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
W = nan(N, T);
M = nan(N, T);
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
    Mt1 = [M(:,t), M(:,1)];
    Wt1 = [W(:,t), W(:,1)];
    dWt = a1 .*  max(Mt1, [], 2) + b1 .* (D(:,t) - d1);
    dMt = ( a2 .* max(Wt1, [], 2) - b2 .* (D(:,t) - d2) ).* ( M(:,t) > 0 );
    W(:, t+1) = W(:, t) + dWt;
    M(:, t+1) = M(:, t) + dMt;
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
