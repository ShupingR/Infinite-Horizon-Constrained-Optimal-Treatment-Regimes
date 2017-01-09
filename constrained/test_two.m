%-------------------------------------------------------------------- 
% replicate the simulated dataset using ode modeling in reinforcement 
% learning clinical design. 
% ref: Reinforcement learning design for cancer clinical trials
% Stat Med. 2009 November 20; 28(26): 3294?3315. doi:10.1002/sim.3720.
% Oct 17
%--------------------------------------------------------------------
clearvars;
nk = 50*5 + 1;
ns = 40;
npar = 20;
%rng(222,'twister');

%%---------%%
% Training %s
%%---------%%
seed_train = 111;
%% Generate data
sample = sample_collect(1000, 7, 111);

%% solve constraint optimization
% Create the kappaList for the constraint
%profile on;
% obj_val = nan(1000,1);
% for t = 1: 1000
%      rng(t);
%      tau0 = rand(25, 1);
%      obj_val(t) = objective_function( tau0, sample_pos);
% end


%kappa = 50;  % ok slightly smaller than fminunc
%kappa = 20; % wont converge after 30min +
kappa_list = linspace( 24 , 74, nk);
fval_list = nan(length(kappa_list), 1);
options = optimset('Algorithm','interior-point', 'LargeScale', 'on',...
                          'PlotFcns',@optimplotfval,'Display','iter');
                      %, 'FinDiffRelStep', 1e-2);
%obj =  @(tau) objective_function_two( tau, sample_pos); % return its negative
tau0 = rand(6, 1);
%options = optimset('PlotFcns',@optimplotfval,'Display','iter', 'FinDiffRelStep', 1e-2);
%[tau_max_vpos, max_vpos, exitflag] = fminunc(obj, tau0, options);
%----------------------------------------------------------------------
% if neither the situation above is satisfied, we solve the problem 
% using constrained optimization nonlinear objective function 
% (negative mean Y to be minimized)
%fileID = fopen('output.txt','a');
%fprintf(fileID,'k, kappa, fval, exitlfag, tau1, tau2, tau3, tau4, tau5, tau6 \r\n');
parpool(npar)
tic;
parfor k = 1:nk
    mytime = cputime;
    kappa = kappa_list(k)
    my_objective = @(tau) objective_function_two( tau, sample);
    my_constraint = @(tau) constraint_function_two( tau, sample, kappa );
    problem = createOptimProblem('fmincon', 'objective', my_objective, ...
                                             'x0', tau0,  'nonlcon', my_constraint, 'options', options);
   
    ms = MultiStart('StartPointsToRun', 'all', 'Display','on');
% 
    [tauSol, fval, exitflag] = run(ms, problem, ns);
    objective_val = -1 * fval;
    A =[ k, kappa, objective_val, exitflag,vec2mat(tauSol, length(tauSol)) ] ;
    fileID = fopen('output_3.txt','a');
    fprintf(fileID,'%d, %4.4f, %4.4f, %d, %4.4f, %4.4f,%4.4f, %4.4f, %4.4f, %4.4f \r\n',A);
end
etime = toc;
timeID = fopen('output_time.txt','a');
fprintf(timeID,' time: %10.2f',etime);
delete(gcp('nocreate'))
%profile viewer
%profsave
