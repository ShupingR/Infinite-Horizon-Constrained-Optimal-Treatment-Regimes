%--------------------------------------------------------------------
% replicate the simulated dataset using ode modeling in reinforcement
% learning clinical design.
% ref: Reinforcement learning design for cancer clinical trials
% Stat Med. 2009 November 20; 28(26): 3294?3315. doi:10.1002/sim.3720.
% Oct 17
%--------------------------------------------------------------------
clearvars;

%rng(222,'twister');

%%---------%%
% Training %s
%%---------%%
seed_train = 111;
seed_test = 222;
%% Generate data
K = 10;
N = 1000; % N = 1000 patients
T = 7; % t = 0, 1,..., T-1, T = 6 => t = 1, 2, ..., 6, 7 months
show = 0; %pic
sample = data_generation( N, T, seed_train, show );

%% policy search + policy evaluation
% choose the center and sigma for features

nb = 1000; % 5 nearest to calculate decision

% center is TK by 2 , sigma is TK
% later modify this aciton groups, return is the total phi dimension TK
center_sig_mat = hyperparm( sample, K, nb, seed_train );

% action that transform state and next state into feature state and next
% feature state

% transfer all the states into feature states in the sample pair
sample = feature_construct( sample, center_sig_mat, K );
display('train set generated');
%%
load tauSolMat
testObjVal = nan(1, 40);
testConstVal = nan(1, 40);
K = 10;

for i = 1:40
    this_tau = tauSolMat(i, :)';
    pos_weight = policy_eval(this_tau, sample, K,  1);
    neg_weight = policy_eval(this_tau, sample, K, -1);
    test_sample = test(N, T, this_tau, seed_test, center_sig_mat, K);
    testObjVal(i) = test_valfun(this_tau, test_sample, pos_weight );
    testConstVal(i) = test_valfun(this_tau, test_sample, neg_weight );
end

figure % new figure
ax1 = subplot(2,1,1); % top subplot
ax2 = subplot(2,1,2); % bottom subplot

x = 1:40;
y1 = testObjVal;
y2 = testConstVal;

plot(ax1, x, y1, 'r-o');
title(ax1, 'constrained optimal policy: vpos vs. kappa');
ylabel(ax1, 'vpos');
xlabel(ax1, 'kappa');

plot(ax2, x, y2, 'b-*');
title(ax2, 'constrained optimal policy: vneg vs. kappa');
ylabel(ax2, 'vneg');
xlabel(ax2, 'kappa');
savefig('plot.fig');

p = gcp;
delete(p);
exit;