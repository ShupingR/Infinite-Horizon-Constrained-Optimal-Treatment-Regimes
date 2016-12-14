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
%% Generate data
N = 1000; % N = 1000 patients
T = 7; % t = 0, 1,..., T-1, T = 6 => t = 1, 2, ..., 6, 7 months
show = 0; %pic
sample = data_generation(N, T, seed_train, show);

%% policy search + policy evaluation
% choose the center and sigma for features
K = 10;
nb = 1000; % 5 nearest to calculate decision 

% center is TK by 2 , sigma is TK
% later modify this aciton groups, return is the total phi dimension TK
center_sig_mat = hyperparm( sample, K, nb, seed_train);

% action that transform state and next state into feature state and next
% feature state

% transfer all the states into feature states in the sample pair
sample = feature_construct( sample, center_sig_mat, K );
display('train set generated');



%% solve constraint optimization
% Create the kappaList for the constraint
profile on;

ns = 1;
kappa = 50;
tau0 = ones(K+1,1);
options = optimset('Algorithm','interior-point',...
                          'LargeScale', 'on');
                      % , ... 'FinDiffRelStep', 1e-2);
%----------------------------------------------------------------------
% if neither the situation above is satisfied, we solve the problem 
% using constrained optimization nonlinear objective function 
% (negative mean Y to be minimized)
my_objective = @(tau) objective( tau, sample, K, 1, -1);
my_constraint = @(tau) constraint( tau, sample, K, kappa );

problem = createOptimProblem('fmincon', 'objective', my_objective, ...
               'x0', tau0,  'nonlcon', my_constraint, 'options', options);%

ms = MultiStart('StartPointsToRun', 'all', 'Display','on');

[tauSol, fval, exitflag] = run(ms, problem, ns);

objective_val = -1 * fval;
constraint_val =  objective(tauSol, sample, K, -1, 1);

profile viewer
profsave


