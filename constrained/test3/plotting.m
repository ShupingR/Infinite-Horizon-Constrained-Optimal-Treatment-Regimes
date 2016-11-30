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

%% find the maximial of value function of positive rewards
t = cputime;
tau0 = ones(K+1, 1);
ns = 3; % number of random starts
vpos =  @(tau) objective(tau, sample, K, 1, -1); % return its negative
%options = optimset('PlotFcns',@optimplotfval,'Display','iter');
%[tau_max_vpos, max_vpos, exitflag] = fminsearch(vpos, tau0, options);
opt_fminunc = optimoptions(@fminunc,...
                                       'Display','iter' ,...
                                       'Algorithm','quasi-newton', ...
                                       'FinDiffRelStep', 1e-2);
problem = createOptimProblem('fminunc', 'objective', vpos, ...
                                            'x0', tau0, 'options', opt_fminunc);%
ms = MultiStart('StartPointsToRun', 'all', 'Display','on');
[tau_max_vpos, max_vpos, exitflag_max_vpos] = run(ms, problem, ns);
max_vpos = -1 * max_vpos;
vneg_max_vpos = objective(tau_max_vpos, sample, K, -1, 1 );
maxVposFile =  'max_vpos.txt';

dlmwrite( maxVposFile, ...
              strcat('tauSol maximize positive value function', num2str(tau_max_vpos')), ...
              'delimiter', ' ', '-append');
dlmwrite( maxVposFile, ...
              strcat('maximal positive value function', num2str(max_vpos)), ...  
              'delimiter', ' ', '-append');
dlmwrite( maxVposFile, ...
              strcat('negative value function of maximal vpos', num2str(vneg_max_vpos)), ...
              'delimiter', ' ', '-append');
dlmwrite( maxVposFile, ...
              strcat('exitflag', num2str(exitflag_max_vpos)), ...
              'delimiter', ' ', '-append');

e = cputime-t;
display(strcat('max obj, time --', num2str(e)));
display('------------------------------------------');
% https://www.mathworks.com/help/optim/ug/fminunc.html#moreabout BFGS quasi-newton
 

%% find the minimal of value function of negative rewards
t = cputime;
vneg1 =  @(tau) objective(tau, sample, K, -1, 1); % return its original val
opt_fminunc = optimoptions(@fminunc,...
                                       'Display','iter',...
                                       'Algorithm','quasi-newton', ...
                                       'FinDiffRelStep', 1e-2);
problem = createOptimProblem('fminunc', 'objective', vneg1, ...
                                            'x0', tau0, 'options', opt_fminunc);%
ms = MultiStart('StartPointsToRun', 'all', 'Display','on');
[tau_min_vneg, min_vneg, exitflag] = run(ms, problem, ns);
vpos_min_vneg = objective(tau_min_vneg, sample, K, 1, 1 );

minVnegFile =  'min_vneg.txt';

dlmwrite( minVnegFile, ...
              strcat('tauSol minimize negative value function', num2str(tau_min_vneg')), ...
              'delimiter', ' ', '-append');
dlmwrite( minVnegFile, ...
              strcat('positive value function of minimal vneg', num2str(vpos_min_vneg)), ...  
              'delimiter', ' ', '-append');
dlmwrite( minVnegFile, ...
              strcat('minimal negative value function', num2str(min_vneg)), ...
              'delimiter', ' ', '-append');
dlmwrite( minVnegFile, ...
              strcat('exitflag', num2str(exitflag)), ...
              'delimiter', ' ', '-append');


e = cputime-t;
display(strcat('min constraint, time --', num2str(e)));
display('------------------------------------------');

%% find the maximal of value function of negative rewards
t = cputime;
vneg2 =  @(tau) objective(tau, sample, K, -1, -1 ); % return its original val
opt_fminunc = optimoptions(@fminunc,...
                                       'Display','iter',...
                                       'Algorithm','quasi-newton', ...
                                       'FinDiffRelStep', 1e-2);
problem = createOptimProblem('fminunc', 'objective', vneg2, ...
                                            'x0', tau0, 'options', opt_fminunc);%                  
ms = MultiStart('StartPointsToRun', 'all', 'Display','on');
[tau_max_vneg, max_vneg, exitflag] = run(ms, problem, ns);
vpos_max_vneg = objective(tau_max_vneg, sample, K, 1, 1 );
max_vneg = -1 * max_vneg;
maxVnegFile =  'max_vneg.txt';

dlmwrite( maxVnegFile, ...
              strcat('tauSol maximize negative value function', num2str(tau_max_vneg')), ...
              'delimiter', ' ', '-append');
dlmwrite( maxVnegFile, ...
              strcat('positive value function of max vneg', num2str(vpos_max_vneg)), ...  
              'delimiter', ' ', '-append');
dlmwrite( maxVnegFile, ...
              strcat('maximial negative value function', num2str(max_vneg)), ...
              'delimiter', ' ', '-append');
dlmwrite( maxVnegFile, ...
              strcat('exitflag', num2str(exitflag)), ...
              'delimiter', ' ', '-append');

e = cputime-t;
display(strcat('max constraint, time --', num2str(e)));
display('------------------------------------------');

%% solve constraint optimization
% Create the kappaList for the constraint
diff = abs( max_vneg - min_vneg ) / 38;
kappaList = linspace(min_vneg - diff, max_vneg  + diff, 40);
kappaListFile =  'kappaList.txt';
dlmwrite(kappaListFile, kappaList, '-append');


% output file names
tauTrainFile =  'train_tau_hat.txt';
% objective function value under the solution above
objectiveTrainFile =  'train_objective_val.txt';
% constraint function value under the solution above
constraintTrainFile =  'train_constraint_val.txt';
% exit flag of solving the problem above
exitflagTrainFile =  'train_exitflag.txt';
% options = optimset('PlotFcns', @optimplotfval, 'LargeScale', 'on', ...
%                  'Algorithm','interior-point' , 'FinDiffRelStep', 1e-2);
progressFile = 'progress.txt';
options = optimset('Algorithm','interior-point',...
                          'LargeScale', 'on', ...
                          'FinDiffRelStep', 1e-2);
 tauSolMat = nan(40,1+K);

parpool(20);
parfor  i = 1:40
    totaltime = cputime;
    t = cputime;
    kappa = kappaList(i);
    %----------------------------------------------------------------------   
    % if minimal vneg > kappa, infeasible
    if (min_vneg > kappa) 
        tauSol = nan(1+K, 1); 
        objective_fval = nan;
        constraint_val = nan;
        exitflag = nan;
        
        tauSolMat(i,:) = tauSol';
        
        dlmwrite( tauTrainFile, ...
                      strcat('kappa', num2str(kappa), ',' , num2str(tauSol')), ...
                      'delimiter', ' ', '-append' );
        dlmwrite( objectiveTrainFile, ...
                      strcat('kappa', num2str(kappa), ',' ,num2str(objective_fval)), ...
                      'delimiter', ' ', '-append' );
        dlmwrite( constraintTrainFile, ...
                      strcat('kappa', num2str(kappa), ',' ,num2str(constraint_val)), ...
                      'delimiter', ' ', '-append' );
        dlmwrite( exitflagTrainFile, ...
                      strcat('kappa', num2str(kappa), ',' , num2str(exitflag)), ...
                      'delimiter', ' ', '-append');
        
        continue;
    end
    
    %----------------------------------------------------------------------   
    % check if the correspond meanZ of maxMeanY satisfies the constraint,
    % aka <= kappa, if it satisfies, then this is the solution
    if (vneg_max_vpos <= kappa) 
        % reach = reach + 1;
        tauSol = tau_max_vpos;
        objective_fval = max_vpos;
        constraint_val = vneg_max_vpos;
        exitflag = exitflag_max_vpos;
        %meanYZ = testset(testseed, tauSol1, tauSol2);
        %meanYtestdList(k) = meanYZ(1);
        %meanZtestdList(k) = meanYZ(2);
        
        tauSolMat(i,:) = tauSol';
        
        dlmwrite( tauTrainFile, ...
                      strcat('kappa', num2str(kappa), ',' , num2str(tauSol')), ...
                      'delimiter', ' ', '-append' );
        dlmwrite( objectiveTrainFile, ...
                      strcat('kappa', num2str(kappa), ',' ,num2str(objective_fval)), ...
                      'delimiter', ' ', '-append' );
        dlmwrite( constraintTrainFile, ...
                      strcat('kappa', num2str(kappa), ',' ,num2str(constraint_val)), ...
                      'delimiter', ' ', '-append' );
        dlmwrite( exitflagTrainFile, ...
                      strcat('kappa', num2str(kappa), ',' , num2str(exitflag)), ...
                      'delimiter', ' ', '-append');

        continue;
    end
    
    %----------------------------------------------------------------------
    % if neither the situation above is satisfied, we solve the problem 
    % using constrained optimization nonlinear objective function 
    % (negative mean Y to be minimized)
    my_objective = @(tau) objective( tau, sample, K, 1, -1);
    my_constraint = @(tau) constraint( tau, sample, K, kappa );
    
    problem = createOptimProblem('fmincon', 'objective', my_objective, ...
                   'x0', tau_min_vneg,  'nonlcon', my_constraint, 'options', options);%
      
    ms = MultiStart('StartPointsToRun', 'all', 'Display','on');
    
    [tauSol, fval, exitflag] = run(ms, problem, ns);
    
    objective_val = -1 * fval;
    constraint_val =  objective(tauSol, sample, K, -1, 1);
    tauSolMat(i,:) = tauSol';
    
    dlmwrite( tauTrainFile, ...
                  strcat('kappa', num2str(kappa), ',' , num2str(tauSol')), ...
                  'delimiter', ' ', '-append' );
    dlmwrite( objectiveTrainFile, ...
                  strcat('kappa', num2str(kappa), ',' ,num2str(objective_val)), ...
                  'delimiter', ' ', '-append' );
    dlmwrite( constraintTrainFile, ...
                  strcat('kappa', num2str(kappa), ',' ,num2str(constraint_val)), ...
                  'delimiter', ' ', '-append' );
    dlmwrite( exitflagTrainFile, ...
                  strcat('kappa', num2str(kappa), ',' , num2str(exitflag)), ...
                  'delimiter', ' ', '-append');
    
    e = cputime-t;
    display(strcat('fmincon, time --', num2str(e)));
    display('------------------------------------------');
    %----------------------------------------------------------------------
    % apple the estimated optimal regime / tauSol above to test dataset

%     meanYZ = testset(testseed, tauSol1, tauSol2);
%     meanYtestdList(k) = meanYZ(1);
%     meanZtestdList(k) = meanYZ(2);
    totale = cputime - totaltime;
    dlmwrite(progressFile, strcat('progress', num2str(i), '. a kappa time:', num2str(totale)) ); 
end
%toc;
% plot 
dlmwrite('tauSolMat', tauSolMat, '-append', 'delimiter', ',');
%%-------%%
% Testing %
%%-------%%
testObjVal = nan(1, 40);
testConstVal = nan(1, 40);

for i = 1:40
    this_tau = tauSolMat(i, :)';
    test_sample = test(N, T, this_tau, seed, center_sig_mat, K);
    testObjVal(i) = objective(this_tau, test_sample, K, 1, 1 );
    testConstVal(i) = objective(this_tau, test_sample, K, -1, 1 );
end

figure % new figure
ax1 = subplot(2,1,1); % top subplot
ax2 = subplot(2,1,2); % bottom subplot

x = kappaList;
y1 = testObjVal;
y2 = testConstVal;

plot(ax1, x, y1, 'r-o');
hline1 = refline(ax1, [0 max_vpos] );
hline1.Color = 'g';
title(ax1, 'constrained optimal policy: vpos vs. kappa');
ylabel(ax1, 'vpos');
xlabel(ax1, 'kappa');

plot(ax2, x, y2, 'b-*');
hline2 = refline(ax2, [0 vneg_max_vpos] );
hline2.Color = 'g';
hline3 = refline(ax2, [0 min_vneg] );
hline3.Color = 'g';
hline4 = refline(ax2, [0 max_vneg] );
hline4.Color = 'g';
title(ax2, 'constrained optimal policy: vneg vs. kappa');
ylabel(ax2, 'vneg');
xlabel(ax2, 'kappa');
savefig('plot.fig');

p = gcp;
delete(p);
exit;