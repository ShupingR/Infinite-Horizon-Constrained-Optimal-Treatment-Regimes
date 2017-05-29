%-------------------------------------
% plot for tau confidence interval
%-------------------------------------
%% add path 
curr_folder = pwd;
addpath(genpath(curr_folder))
%% load data
% cd ~/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/scripts/sim/
REP = 300;
test_pos_val_mat = nan(REP, 20);
test_neg_val_mat = nan(REP, 20);
pos_weight_mat = nan(REP, 20, 5*5);
neg_weight_mat = nan(REP, 20, 5*5);
min_constraint_mat = nan(REP, 20);
max_constraint_mat = nan(REP, 20);
min_objective_mat = nan(REP, 20);
max_objective_mat = nan(REP, 20);
tau_mat = nan( 20, 6, REP);
% test samples
K = 4; % number of radial basis functions, not include the intercept
L = 5; % number of dosage levels
N = 7000; % training set sample size
T = 7; % number of stages
discount = 0.8;
test_seed = 111;
rng(test_seed,'twister');
test_sample = sample_collect(N, T, K, test_seed); % generate training set

for rep = 1:REP
    % constrained par
    % cd ~/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/sim_results/constrained/
    fileName = [ 'output_may_16_constrained_sequential_initial_rep_' num2str(rep) ]; 
    dataStruct.(fileName) =  load( [ fileName '.txt' ]);
    dat = sortrows(dataStruct.(fileName) , 1); % sort the result by first column
    dat = array2table(dat);
    dat.Properties.VariableNames = {'k' 'nu' 'objective_val' 'constraint_val' 'exitflag' ...
                                                  'tau0' 'tau1' 'tau2' 'tau3' 'tau4' 'tau5'};
    tauTab = [ dat.tau0, dat.tau1, dat.tau2, dat.tau3, dat.tau4, dat.tau5 ];
    nuList = dat.nu; % constraint value on secondary value 
    
    %% generate train dataset 
    % cd ~/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/scripts/sim/
    which_reward_pos = 1; % positive reward
    which_reward_neg = -1; % negative reward
    sign = 1; % original function value 
    % plot pareto efficient frontier on training dataset
    for k = 1:length(nuList)
        tic;
        tau = tauTab(k, :)';
        [ test_pos_val_mat(rep, k), pos_weight_mat(rep, k, :) ] = ...
            value_function(tau, test_sample, discount, K, L, which_reward_pos, sign);
        [ test_neg_val_mat(rep, k), neg_weight_mat(rep, k, :) ] = ...
            value_function(tau, test_sample, discount, K, L, which_reward_neg, sign);
        tau_mat(k, :, rep) = tau';
        toc;
    end
    % unconstrained part
    % cd ~/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/sim_results/unconstrained/
    fileName2 = [ 'output_may_16_unconstrained_rep_' num2str(rep) ]; 
    dataStruct2.(fileName2) =  load( [ fileName2 '.txt' ]);
    dat2 = dataStruct2.(fileName2);
    min_constraint_mat(rep, :) = repmat(dat2(1, 2), 1, length(nuList));
    max_constraint_mat(rep, :) = repmat(dat2(2, 2), 1, length(nuList));
    min_objective_mat(rep, :) = repmat(dat2(3, 2), 1, length(nuList));
    max_objective_mat(rep, :) = repmat(dat2(4, 2), 1, length(nuList));
end

%%
% pos
mean_test_pos_val = mean(test_pos_val_mat, 1);
std_test_pos_val = std(test_pos_val_mat);
 % 95% confidence interval 
upper_ci_test_pos_val = mean_test_pos_val + 1.96 * std_test_pos_val / sqrt(REP);
lower_ci_train_pos_val = mean_test_pos_val - 1.96 * std_test_pos_val / sqrt(REP);
neg_ci_pos_val = mean_test_pos_val - lower_ci_train_pos_val;
pos_ci_pos_val = upper_ci_test_pos_val - mean_test_pos_val;

% neg
mean_test_neg_val = mean(test_neg_val_mat, 1);
std_test_neg_val = std(test_neg_val_mat);
 % 95% confidence interval 
upper_ci_test_neg_val = mean_test_neg_val + 1.96 * std_test_neg_val / sqrt(REP);
lower_ci_train_neg_val = mean_test_neg_val - 1.96 * std_test_neg_val / sqrt(REP);
neg_ci_neg_val = mean_test_neg_val - lower_ci_train_neg_val;
pos_ci_neg_val = upper_ci_test_neg_val - mean_test_neg_val;

% unconstrained value
% min constraint
mean_min_constraint = mean(min_constraint_mat, 1);
std_min_constraint = std(min_constraint_mat);
upper_ci_min_constraint = mean_min_constraint + 1.96 * std_min_constraint / sqrt(REP);
lower_ci_min_constraint = mean_min_constraint - 1.96 * std_min_constraint / sqrt(REP);

% max constraint
mean_max_constraint = mean(max_constraint_mat, 1);
std_max_constraint = std(max_constraint_mat);
upper_ci_max_constraint = mean_max_constraint + 1.96 * std_max_constraint / sqrt(REP);
lower_ci_max_constraint = mean_max_constraint - 1.96 * std_max_constraint / sqrt(REP);

% min objective
mean_min_objective = mean(min_objective_mat, 1);
std_min_objective = std(min_objective_mat);
upper_ci_min_objective = mean_min_objective + 1.96 * std_min_objective / sqrt(REP);
lower_ci_min_objective = mean_min_objective - 1.96 * std_min_objective / sqrt(REP);

% max objective
mean_max_objective = mean(max_objective_mat, 1);
std_max_objective = std(mean_max_objective);
upper_ci_max_objective = mean_max_objective + 1.96 * std_max_objective / sqrt(REP);
lower_ci_max_objective = mean_max_objective - 1.96 * std_max_objective / sqrt(REP);

mean_tau = mean(tau_mat, 3);
std_tau = std(tau_mat, 0, 3);
mean_std_tau = nan(20, 12);
mean_std_tau(:,1:2:end) = mean_tau;
mean_std_tau(:, 2:2:end) = std_tau;
upper_ci_tau = mean_tau + 1.96 * std_tau / sqrt(REP);
lower_ci_tau = mean_tau - 1.96 * std_tau / sqrt(REP);
ci_tau_tab= horzcat( nuList, mean_test_pos_val', std_test_pos_val', ...
                 mean_test_neg_val', std_test_neg_val', mean_std_tau);
             
cd ./plot_results/
% plot
width=10;
height=16;
x0=1;
y0=1;
figure('Units','inches', 'Position',[x0 y0 width height], 'PaperPositionMode','auto');
% (x0,y0) = position of the lower left side of the figure

for i = 1: 6
   subplot(6,1,i)
   errorbar(nuList, mean_tau(:,i), upper_ci_tau(:,i), lower_ci_tau(:,i), 'o');
   str = sprintf('$$\\widehat{\\tau}_{\\nu,%d}$$',i);
   %text('Interpreter','latex','Position',[1 2],'String',str)
    % ylabel({'$\widehat{\tau}_{\nu,%d}$',i}, 'interpreter' ,'latex', 'FontSize',15 )
   txt = ['$\widehat{\tau}_{\nu},$', num2str(i)];
   ylabel({txt}, 'interpreter' ,'latex', 'FontSize',15)
  % tex file
end
xlabel({'Constraints $\nu$'}, 'interpreter' ,'latex', 'FontSize',15 )
% ylabel({'$\widehat{V}$ values of estimated constrained optimal regimes'},...
%           'interpreter' ,'latex', 'FontSize',15 )
currentFigure = gcf;
title(currentFigure.Children(end), {'Confidence interval for components of $\widehat{\bf{\tau}}_{\nu}$'},'interpreter' ,'latex' , 'FontSize',15);
%title({'Confidence interval for $\widehat{\boldsymbol{\tau}}'},'interpreter' ,'latex', 'FontSize',15);
% legend({'$\widehat{V}_1$ vs. $\nu$',  '$\widehat{V}_2$ vs. $\nu$'}, ...
%          'interpreter' ,'latex', 'Location','SouthEast','FontSize',15);

% axis([0 t(end) -1.5 1.5]);
set(gca, 'Units','normalized', ...
     'FontUnits','points',...
     'FontWeight','normal')%,...
    % 'FontSize',15);
print('tau_ci', '-dpdf', '-bestfit' ) ;

