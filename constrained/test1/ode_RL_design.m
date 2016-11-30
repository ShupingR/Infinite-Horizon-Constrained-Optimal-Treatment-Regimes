%-------------------------------------------------------------------- 
% replicate the simulated dataset using ode modeling in reinforcement 
% learning clinical design. 
% ref: Reinforcement learning design for cancer clinical trials
% Stat Med. 2009 November 20; 28(26): 3294?3315. doi:10.1002/sim.3720.
% Oct 17
%--------------------------------------------------------------------
clearvars;
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

%% Generate data
N = 1000; % N = 1000 patients
T = 7; % t = 0, 1,..., T-1, T = 6 => t = 1, 2, ..., 6, 7 months
show = 1; %pic
sample = data_generation(N, T, show);

%% Generate grid for discritization
%%%% order wellness and tumorsize
M_x = 1000;
M_y = 1000;
grid = gen_grid(sample(1).low_wellness, sample(1).upp_wellness,...
                sample(1).low_tumorsize, sample(1).upp_tumorsize,...
                M_x, M_y);

%% Project each sample onto the nearest grid, current state and next state
i = 1;
proj_sample = sample;
for each_sample = sample
    [ closest_index_current, closest_index_next ] = ...
        nearest_vertx(grid, each_sample);
    proj_sample(i).wellness = grid(closest_index_current, 1);
    proj_sample(i).tumorsize = grid(closest_index_current, 2);
    proj_sample(i).next_wellness = grid(closest_index_next, 1);
    proj_sample(i).next_tumorsize = grid(closest_index_next, 2);
    i = i + 1;
end
% Do not re-run above,  save the matrix
%% policy search + policy evaluation
K = 5;
TK = K * 2 * 6;
P = 5; % 5 nearest to calculate decision 
tau0 = 0.5;
%tau0 = zeros(1, TK);
% tau0(2) = 0.5;
% tau0(4) = -0.5;
% tau0(8) = -1;
seed = 222;
kappa = 40;
ns = 5;
[ center, sigma2 ] = hyperparm( sample, TK, P, seed );

all_proj_phi = all_feature( proj_sample, center, sigma2, TK );
                  
my_objective = @(tau) objective( proj_sample, all_proj_phi, center, ...
                                 sigma2, tau, K );

%my_constraint = @(tau) constraint( proj_sample, all_proj_phi, center,...
%                                   sigma2, tau, K, kappa );

options = optimset('PlotFcns', @optimplotfval);
tauSol_fminsearch = fminsearch(my_objective, tau0, options);

% opts = optimoptions(@fminsearch,'Display','iter-detailed',...
%       'Algorithm','interior-point'); % , 'FinDiffRelStep', 1e-2);  

% opts = optimoptions(@fmincon,'Display','iter-detailed',...
%       'Algorithm','interior-point'); % , 'FinDiffRelStep', 1e-2);  

% problem = createOptimProblem('fmincon', 'objective', my_objective, ...
%          'x0', tau0,  'nonlcon', my_constraint, 'options', opts);

% ms = MultiStart('StartPointsToRun', 'all', 'Display','off');

% [tauSol, fval, exitflag] = run(ms, problem, ns);

% obj_val = -fval;


%%
% remember to set seed for simulation to avoid stochastic 
% fimsearch use nealder-mean simplex method, gradient free

% 
% Multiple Start ininital point
% tau0 = [-0.5, 0.5, 0.5, -0.5 ];
% 
% display('Z')
% find min Z for later check if it satisfies the constraint, aka <= kappa
% tic;
% objZ = @(tau) preEstPrY (tau, Z, H2, A2, H1, A1, n, 1) ;
% optsZ = optimoptions(@fminunc,'Algorithm', 'quasi-newton', 'Display','off' , 'FinDiffRelStep', 1e-2);
% problemZ = createOptimProblem('fminunc', 'objective', objZ, 'x0', tau0,  'options', optsZ);
% msZ = MultiStart('StartPointsToRun', 'all', 'Display','off');
% [tauSolZ, fvalZ, exitflagZ] = run(msZ, problemZ, ns);
% toc;
% 
% display('Y')
% find max Y for later check if its corresponding Z satisfies the constraint, aka <= kappa
% tic;
% objY = @(tau) preEstPrY (tau, Y, H2, A2, H1, A1, n, -1) ;
% optsY = optimoptions(@fminunc,'Algorithm', 'quasi-newton', 'Display','off' , 'FinDiffRelStep', 1e-2);
% problemY = createOptimProblem('fminunc', 'objective', objY, 'x0', tau0,  'options', optsY);
% msY = MultiStart('StartPointsToRun', 'all', 'Display','off');
% [tauSolY, fvalY, exitflagY] = run(msY, problemY, ns);
% fvalY = -fvalY;
% fvalZmaxY = preEstPrY (tauSolY, Z, H2, A2, H1, A1, n, 1) ;
% toc;

% %%
% display('C')
% % Create the kappaList for the constraint
% stdZ = std(Z);
% kappaList = linspace (min(Z) - 0.5 * stdZ , max(Z) + 0.5 * stdZ, nk);
% kappaListFile =  'test7_kappaList.txt';
% dlmwrite(kappaListFile, kappaList, '-append');
% %reach = 0;
% %profile on
% %write out solutions
% %parpool(2);
%  %tic;
% 
% for k = 1:nk 
%     kappa = kappaList(k);
%     %----------------------------------------------------------------------
%     % output file names
%     optTauHatFile =  strcat('test7_optTauHat_', num2str(kappa), '.txt');
%         % estimated optimal regime indexing parameters
%     optObjValFile =  strcat('test7_optObjVal_', num2str(kappa), '.txt');
%         % objective function value under the solution above
%     optConValFile =  strcat('test7_optContVal_', num2str(kappa), '.txt');
%         % constraint function value under the solution above
%     exitFlagFile =  strcat('test7_exitFlagHat_', num2str(kappa), '.txt');
%         % exit flag of solving the problem above
%     optMeanYTestFile =  strcat('test7_optMeanYTest_', num2str(kappa), '.txt');
%         % estimated meanY under the estimated constrained opt regime
%     optMeanZTestFile =  strcat('test7_optMeanZTest_', num2str(kappa), '.txt');
%         % estimated meanZ under the estimated constrained opt regime
%     %----------------------------------------------------------------------   
%     % check if minMeanZ satisfies the constraint, aka <= kappa
%     % if minMeanZ > kappa, problem infeasible. We skip to the next kappa
%     if (fvalZ > kappa) 
%         tauSol = [ NaN, NaN, NaN, NaN ]; 
%         fval = NaN;
%         con = NaN;
%         exitflag = NaN;
%         meanYtestdList(k) = NaN;
%         meanZtestdList(k) = NaN;
%         dlmwrite(optTauHatFile, tauSol, '-append');
%         dlmwrite(optObjValFile, fval, '-append');
%         dlmwrite(optConValFile, con, '-append');
%         dlmwrite(exitFlagFile, exitflag, '-append');
%         continue;
%     end
%     
%     %----------------------------------------------------------------------   
%     % check if the correspond meanZ of maxMeanY satisfies the constraint,
%     % aka <= kappa, if it satisfies, then this is the solution
%     if (fvalZmaxY <= kappa) 
%         % reach = reach + 1;
%         tauSol = tauSolY;
%         con = fvalZmaxY;
%         exitflag = exitflagY;
%         tauSol1 = tauSol(1:2)';
%         tauSol2 = tauSol(3:4)';
%         meanYZ = testset(testseed, tauSol1, tauSol2);
%         meanYtestdList(k) = meanYZ(1);
%         meanZtestdList(k) = meanYZ(2);
%  
%         dlmwrite(optTauHatFile, tauSol, '-append');
%         dlmwrite(optObjValFile, fvalY, '-append');
%         dlmwrite(optConValFile, con, '-append');
%         dlmwrite(exitFlagFile, exitflag, '-append');
%         
%         % if maxY is achieved for three times, we stop calculating further,
%         % not suitable for parfor
%         % if ( reach > 3) 
%         %    break;
%         % end
%         %
%         continue;
%     end
%     
%     %----------------------------------------------------------------------
%     % if neither the situation above is satisfied, we solve the problem 
%     % using constrained optimization nonlinear objective function 
%     % (negative mean Y to be minimized)
%     obj = @(tau) preEstPrY (tau, Y, H2, A2, H1, A1, n, -1) ;
%     %Nonlinear Inequality and Equality Constraints
%     constraint = @(tau) preEstPrZ(tau, Z, H2, A2, H1, A1, kappa, n);
%     % fmincon options
%     % options = optimoptions(@fmincon,'Display','iter-detailed','Algorithm','interior-point' , 'FiniteDifferenceStepSize', 1e-2);  
%     opts = optimoptions(@fmincon,'Display','iter-detailed','Algorithm','interior-point' , 'FinDiffRelStep', 1e-2);  
%     % opts = optimset('Algorithm', 'interior-point', 'FinDiffRelStep',1e-2); 
%     problem = createOptimProblem('fmincon', 'objective', obj, 'x0', tau0,  'nonlcon', constraint, 'options', opts);
%     ms = MultiStart('StartPointsToRun', 'all', 'Display','off');
%     [tauSol, fval, exitflag] = run(ms, problem, ns);
%     con = preEstPrY (tauSol, Z, H2, A2, H1, A1, n, 1) ;
%     fval = -fval;
%     dlmwrite(optTauHatFile, tauSol, '-append');
%     dlmwrite(optObjValFile, fval, '-append');
%     dlmwrite(optConValFile, con, '-append');
%     dlmwrite(exitFlagFile, exitflag, '-append');
% 
%     %----------------------------------------------------------------------
%     % apple the estimated optimal regime / tauSol above to test dataset
%     tauSol1 = tauSol(1:2)';
%     tauSol2 = tauSol(3:4)';
%     meanYZ = testset(testseed, tauSol1, tauSol2);
%     meanYtestdList(k) = meanYZ(1);
%     meanZtestdList(k) = meanYZ(2);
% end
%toc;
% plot 
