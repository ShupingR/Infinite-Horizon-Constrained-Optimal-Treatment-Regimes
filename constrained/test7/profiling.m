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
[ sample_pos, sample_neg ] = train_sample_collect(1000, 7, 111);

%% solve constraint optimization
% Create the kappaList for the constraint
profile on;

ns = 1;
kappa = 50;
tau0 = rand(25, 1);
% options = optimset('Algorithm','interior-point', 'LargeScale', 'on');
obj =  @(tau) objective_function( tau, sample_pos); % return its negative
options = optimset('PlotFcns',@optimplotfval,'Display','iter');
[tau_max_vpos, max_vpos, exitflag] = fminunc(obj, tau0, options);
                      % , ... 'FinDiffRelStep', 1e-2);
%----------------------------------------------------------------------
% if neither the situation above is satisfied, we solve the problem 
% using constrained optimization nonlinear objective function 
% (negative mean Y to be minimized)
% my_objective = @(tau) objective_function( tau, sample_pos);
% my_constraint = @(tau) constraint_function( tau, sample_neg, kappa );
% 
% problem = createOptimProblem('fmincon', 'objective', my_objective, ...
%                'x0', tau0,  'nonlcon', my_constraint, 'options', options);%
% 
% ms = MultiStart('StartPointsToRun', 'all', 'Display','on');
% 
% [tauSol, fval, exitflag] = run(ms, problem, ns);
% 
% objective_val = -1 * fval;

profile viewer
profsave

