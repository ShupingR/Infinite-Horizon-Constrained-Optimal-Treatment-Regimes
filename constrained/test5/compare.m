% compare weights computed by Sherman-Morrison formula and original LSTDq
N = 1000;
T = 7;
k = 9;
train_seed = 111;
show = 0;
gamma = 0.8;
which_reward = 1;
train_sample = train_dat_gen(N, T, k, train_seed, show);
lspi_orig( train_sample, k, gamma,  which_reward)
% ns = 3; % number of random starts
% vpos =  @(tau) objective(tau, sample, K, 1, -1); % return its negative
% options = optimset('PlotFcns',@optimplotfval,'Display','iter');
% [tau_max_vpos, max_vpos, exitflag] = fminsearch(vpos, tau0, options);