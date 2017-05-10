clear;
%profile on; 
% Training to get q function's weight and policy index
tic;
nk = 20; % number of bounds
npar = nk; % parallel number 
ns = 5; % number of random start
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
fileName0 = 'output_may_10_unconstrained.txt';
sign_min = 1; % to minimize a function
sign_max = -1; % to maximize a function
pos_reward = 1; % calculate wrt positive reward
neg_reward = -1; % calculate wrt negative reward

% 
tau0 = [ 0; 0; 0; 0; 0; 0 ]; 
lb = [ -5; -5; -5; -5; -5; -5 ]; 
ub = [ 5; 5; 5; 5; 5; 5 ];

options = optimset('Algorithm','interior-point', 'LargeScale', 'off', ...
                          'Display','off');
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

%% solve min/max constraint
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
nu_list = linspace(constraint_min , constraint_max, nk); % a range of constraint
tau0 = tau_constraint_max;
%for rep = 1:1
    %seed = rep + 10;
    fileName = strcat('output_may_10_constrained.txt');
    %rng(seed,'twister');
    %sample = sample_collect(N, T, K, seed); % generate training set
    parfor k = 1:nk
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
    end
   % fprintf(fileID,'%d, %4.4f, %4.4f, %d, %4.4f, %4.4f,%4.4f, %4.4f, %4.4f, %4.4f \r\n',A);
%end
delete(gcp('nocreate'));
toc;
fclose('all');
%profsave;
%profile viewer
quit force;
