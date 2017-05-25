% initial start for K is the solution of K-1
% second batch of 20 replicates
clear;
tic;
nk = 20; % number of bounds
npar = 20; % parallel number
ns = 50; % number of random start
K = 4; % number of radial basis functions, not include the intercept
L = 5; % number of dosage levels
N = 7000; % training set sample size
T = 7; % number of stages
discount = 0.8;
seed = 222;
rng(seed,'twister');
sample = sample_collect(N, T, K, seed); % generate training set
% mu parm for radial basis functions

%% set up the solver
fileName0 = 'output_may_16_unconstrained_batch_2.txt';
sign_min = 1; % to minimize a function
sign_max = -1; % to maximize a function
pos_reward = 1; % calculate wrt positive reward
neg_reward = -1; % calculate wrt negative reward

% 
tau0 = [ 0; 0; 0; 0; 0; 0 ]; % initial
lb = [ -5; -5; -5; -5; -5; -5 ]; % lower bounds on tau
ub = [ 5; 5; 5; 5; 5; 5 ]; % upper bounds on tau

options = optimset('Algorithm','interior-point', 'LargeScale', 'off', 'Display','off');
                      %,'FinDiffRelStep', 1e-4, ...
                       %   'PlotFcns',@optimplotfval);
ms = MultiStart('StartPointsToRun', 'all', 'Display','off');

%% solve min/max constraint
% find min constraint fun value
my_constraint_min = @(tau) value_function( tau, sample, discount, K, L, ...
                                neg_reward, sign_min );
problem = createOptimProblem('fmincon', 'objective', my_constraint_min, ...
                                             'x0', tau0, 'lb',lb,'ub', ub, 'options', options);
[tau_constraint_min, fval, exitflag] = run(ms, problem, ns);
constraint_min = sign_min*fval;
A = [ 1, constraint_min, exitflag, ...
        vec2mat(tau_constraint_min, length(tau_constraint_min)) ];
dlmwrite(fileName0, A, '-append');
    
%%
my_constraint_max = @(tau) value_function( tau, sample, discount, K, L, ...
                                neg_reward, sign_max );
problem = createOptimProblem('fmincon', 'objective', my_constraint_max, ...
                                            'x0', tau0, 'lb',lb,'ub', ub, 'options', options);
[tau_constraint_max, fval, exitflag] = run(ms, problem, ns);
constraint_max = sign_max*fval;
A = [ 2, constraint_max, exitflag, ...
        vec2mat(tau_constraint_max, length(tau_constraint_max)) ];
dlmwrite(fileName0, A, '-append');

%% solve min/max objective
% find min objective fun value
my_objective_min = @(tau) value_function( tau, sample, discount, K, L, ...
                               pos_reward, sign_min );
problem = createOptimProblem('fmincon', 'objective', my_objective_min, ...
                                            'x0', tau0, 'lb',lb,'ub', ub, 'options', options);
[tau_objective_min, fval, exitflag] = run(ms, problem, ns);
objective_min = sign_min*fval;
A = [ 3, objective_min, exitflag, ...
        vec2mat(tau_objective_min, length(tau_objective_min)) ];
dlmwrite(fileName0, A, '-append');

%%
% find max objective fun value
my_objective_max = @(tau) value_function( tau, sample, discount, K, L, ...
                                pos_reward, sign_max );
problem = createOptimProblem('fmincon', 'objective', my_objective_max,...
                                            'x0', tau0, 'lb',lb,'ub', ub, 'options', options);
[tau_objective_max, fval, exitflag] = run(ms, problem, ns);
objective_max = sign_max*fval;
A = [ 4, objective_max, exitflag, ...
        vec2mat(tau_objective_max, length(tau_objective_max)) ];
dlmwrite(fileName0, A, '-append');

%%
parpool(npar)    
parfor rep = 181:200
    seed = rep + 1000;
    % unconstrained file name
    fileName0 = strcat('output_may_16_unconstrained_rep_', num2str(rep), '.txt');

    % generate sample replicates
    rng(seed, 'twister');

    sample = sample_collect(N, T, K, seed); % generate training set
        
    tau0 = [ 0; 0; 0; 0; 0; 0 ]; % initial
    lb = [ -5; -5; -5; -5; -5; -5 ]; % lower bounds on tau
    ub = [ 5; 5; 5; 5; 5; 5 ]; % upper bounds on tau

    options = optimset('Algorithm','interior-point', 'LargeScale', 'off', 'Display','off');
    ms = MultiStart('StartPointsToRun', 'all', 'Display','off');

    % find min constraint fun value
    my_constraint_min = @(tau) value_function( tau, sample, discount, K, L, ...
                                    neg_reward, sign_min );
    problem = createOptimProblem('fmincon', 'objective', my_constraint_min, ...
                                                 'x0', tau0, 'lb',lb,'ub', ub, 'options', options);
                                             
    [tauSol, fval, exitflag] = run(ms, problem, ns);
    constraint_min_rep = sign_min*fval;
    A = [ 1, constraint_min_rep, exitflag, ...
            vec2mat(tauSol, length(tauSol)) ];
    dlmwrite(fileName0, A, '-append');

    %%
    my_constraint_max = @(tau) value_function( tau, sample, discount, K, L, ...
                                    neg_reward, sign_max );
    problem = createOptimProblem('fmincon', 'objective', my_constraint_max, ...
                                                'x0', tau0, 'lb',lb,'ub', ub, 'options', options);
    [tauSol, fval, exitflag] = run(ms, problem, ns);
    constraint_max_rep = sign_max*fval;
    A = [ 2, constraint_max_rep, exitflag, ...
            vec2mat(tauSol, length(tauSol)) ];
    dlmwrite(fileName0, A, '-append');

    %% solve min/max constraint
    % find min objective fun value
    my_objective_min_rep = @(tau) value_function( tau, sample, discount, K, L, ...
                                   pos_reward, sign_min );
    problem = createOptimProblem('fmincon', 'objective', my_objective_min_rep, ...
                                                'x0', tau0, 'lb',lb,'ub', ub, 'options', options);
    [tauSol, fval, exitflag] = run(ms, problem, ns);
    objective_min_rep = sign_min*fval;
    A = [ 3, objective_min_rep, exitflag, ...
            vec2mat(tauSol, length(tauSol)) ];
    dlmwrite(fileName0, A, '-append');

    %%
    % find max objective fun value
    my_objective_max_rep = @(tau) value_function( tau, sample, discount, K, L, ...
                                    pos_reward, sign_max );
    problem = createOptimProblem('fmincon', 'objective', my_objective_max_rep,...
                                                'x0', tau0, 'lb',lb,'ub', ub, 'options', options);
    [tauSol, fval, exitflag] = run(ms, problem, ns);
    objective_max_rep = sign_max*fval;
    A = [ 4, objective_max_rep, exitflag, ...
            vec2mat(tauSol, length(tauSol)) ];
    dlmwrite(fileName0, A, '-append');
    
    % constrained file name
    fileName = strcat('output_may_16_constrained_sequential_initial_rep_', num2str(rep), '.txt');
    nu_list = linspace(constraint_min , constraint_max, nk); % a range of constraint
    for k = 1:nk
        nu = nu_list(k);
        my_objective = @(tau) value_function(tau, sample, discount, K, L, pos_reward, sign_max);
        my_constraint = @(tau) constraint(tau, sample, discount, K, L, nu) ;
         % if min can not be less than upper bound skip
        options = optimset('Algorithm','interior-point', 'LargeScale', 'on', 'Display','off');
                                   %   'PlotFcns',@optimplotfval,);
        problem = createOptimProblem('fmincon', 'objective', my_objective, ...
                           'x0', tau0, 'lb',lb,'ub', ub,  'nonlcon', my_constraint, 'options', options);

        ms = MultiStart('StartPointsToRun', 'all', 'Display','off');
        % 
        [tauSol, fval, exitflag] = run(ms, problem, ns);
        objective_val = -1 * fval;
        constraint_val = value_function( tauSol, sample, discount, K, L, neg_reward, sign_min);
        A =[ k, nu, objective_val, constraint_val, exitflag,vec2mat(tauSol, length(tauSol)) ] ;
        dlmwrite(fileName, A, '-append')
        tau0 = tauSol;
    end
   % fprintf(fileID,'%d, %4.4f, %4.4f, %d, %4.4f, %4.4f,%4.4f, %4.4f, %4.4f, %4.4f \r\n',A);
end
delete(gcp('nocreate'));
toc;
fclose('all');
%profsave;
%profile viewer
quit force;
