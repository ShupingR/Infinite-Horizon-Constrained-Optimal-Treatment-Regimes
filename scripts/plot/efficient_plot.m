%-----------------
% efficient plot 
%-----------------
addpath(genpath('/Users/shuping.ruan/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/'));
%% load data
REP = 3;
test_pos_val_mat = nan(REP, 20);
test_neg_val_mat = nan(REP, 20);
pos_weight_mat = nan(REP, 20, 5*5);
neg_weight_mat = nan(REP, 20, 5*5);
min_constraint_mat = nan(REP, 20);
max_constraint_mat = nan(REP,20);
min_objective_mat = nan(REP, 20);
max_objective_mat = nan(REP, 20);

% test samples
% cd ~/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/scripts/sim/
K = 4; % number of radial basis functions, not include the intercept
L = 5; % number of dosage levels
N = 7000; % training set sample size
T = 7; % number of stages
discount = 0.8;
test_seed = 111;
rng(test_seed,'twister');
% generate test dataset
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

    % cd ~/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/scripts/sim/
    which_reward_pos = 1; % positive reward
    which_reward_neg = -1; % negative reward
    sign = 1; % original function value 
    % plot pareto efficient frontier on test dataset
    for k = 1:length(nuList)
        tic;
        tau = tauTab(k, :)';
        [ test_pos_val_mat(rep, k), pos_weight_mat(rep, k, :) ] = ...
            value_function(tau, test_sample, discount, K, L, which_reward_pos, sign);
        [ test_neg_val_mat(rep, k), neg_weight_mat(rep, k, :) ] = ...
            value_function(tau, test_sample, discount, K, L, which_reward_neg, sign);
        toc;
    end
    % unconstrained part
    % cd ~/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/sim_results/unconstrained/
    fileName2 = [ 'output_may_16_unconstrained_rep_' num2str(rep) ]; 
    dataStruct2.(fileName2) =  load( [ fileName2 '.txt' ]);
    dat2 = dataStruct2.(fileName2);
    % cd ~/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/scripts/sim/
    % min constraint value
    min_constraint_tau = dat2(1, 4:9)';
    min_constraint = value_function(min_constraint_tau, test_sample, discount, K, L, which_reward_neg, sign);
    min_constraint_mat(rep, :) = repmat(min_constraint, 1, length(nuList));
    % max constraint value
    max_constraint_tau = dat2(2, 4:9)';
    max_constraint = value_function(max_constraint_tau, test_sample, discount, K, L, which_reward_neg, sign);
    max_constraint_mat(rep, :) = repmat(max_constraint, 1, length(nuList));
    % min objective value
    min_objective_tau = dat2(3, 4:9)';
    min_objective = value_function(min_objective_tau, test_sample, discount, K, L, which_reward_pos, sign);
    min_objective_mat(rep, :) = repmat(min_objective, 1, length(nuList));
    % max objective value
    max_objective_tau = dat2(4, 4:9)';
    max_objective = value_function(max_objective_tau, test_sample, discount, K, L, which_reward_pos, sign);
    max_objective_mat(rep, :) = repmat(max_objective, 1, length(nuList));
end

%%
cd /Users/shuping.ruan/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/plot_results/
h = figure;
% pos
mean_test_pos_val = mean(test_pos_val_mat, 1);
std_test_pos_val = std(test_pos_val_mat);
 % 95% confidence interval 
err_test_pos_val = 1.96 * std_test_pos_val / sqrt(REP);
errorbar(nuList, mean_test_pos_val, err_test_pos_val, err_test_pos_val,'--ro');
hold on

% neg
mean_test_neg_val = mean(test_neg_val_mat, 1);
std_test_neg_val = std(test_neg_val_mat);
 % 95% confidence interval 
err_test_neg_val = 1.96 * std_test_neg_val / sqrt(REP);
errorbar(nuList, mean_test_neg_val, err_test_neg_val, err_test_neg_val, '--bo');

hline = refline(1,0);
set(hline,'LineStyle',':', 'LineWidth',1.5);

% unconstrained value
% min constraint
mean_min_constraint = mean(min_constraint_mat, 1);
%std_min_constraint = std(min_constraint_mat);
%err_min_constraint = 1.96 * std_min_constraint / sqrt(REP);
%errorbar(nuList, mean_min_constraint, err_min_constraint, err_min_constraint, '--ro');
hline = refline(0, mean_min_constraint(1));
set(hline,'LineStyle',':', 'Color', 'b', 'LineWidth',1.5);

% max constraint
mean_max_constraint = mean(max_constraint_mat, 1);
%std_max_constraint = std(max_constraint_mat);
%err_max_constraint = 1.96 * std_max_constraint / sqrt(REP);
%errorbar(nuList, mean_max_constraint, lower_ci_max_constraint, upper_ci_max_constraint,'--ro');
hline = refline(0, mean_max_constraint(1));
set(hline,'LineStyle',':', 'Color', 'b', 'LineWidth',1.5);

% min objective
mean_min_objective = mean(min_objective_mat, 1);
%std_min_objective = std(min_objective_mat);
%err_min_objective =  1.96 * std_min_objective / sqrt(REP);
%lower_ci_min_objective = 1.96 * std_min_objective / sqrt(REP);
% errorbar(nuList, mean_min_objective, lower_ci_min_objective, upper_ci_min_objective,'--bo');
hline = refline(0, mean_min_objective(1));
set(hline,'LineStyle',':', 'Color', 'r','LineWidth',1.5);

% max objective
mean_max_objective = mean(max_objective_mat, 1);
%std_max_objective = std(mean_max_objective);
%upper_ci_max_objective = mean_max_objective + 1.96 * std_max_objective / sqrt(REP);
%lower_ci_max_objective = mean_max_objective - 1.96 * std_max_objective / sqrt(REP);
% errorbar(nuList, mean_max_objective, lower_ci_max_objective, upper_ci_max_objective,'--bo');
hline = refline(0, mean_max_objective(1));
set(hline,'LineStyle',':', 'Color', 'r','LineWidth',1.5);

xlabel({'Constraints $\nu$ '}, ...
         'interpreter' ,'latex', 'FontSize',15 )
ylabel({'$\widehat{V}^{+/-}$ of estimated constrained optimal regimes'},...
          'interpreter' ,'latex', 'FontSize',15 )
title({'Efficient frontier plot $\widehat{V}^{+/-}$ vs. $\nu$'},...
        'interpreter' ,'latex', 'FontSize',15);
legend({'$\widehat{V}^{+}$ vs. $\nu$',  '$\widehat{V}^{-}$ vs. $\nu$'}, ...
           'interpreter' ,'latex', 'Location','SouthEast','FontSize',15);
%axis([0 t(end) -1.5 1.5]);
set(gca, 'Units','normalized', ...
       'FontUnits','points',... 
       'FontWeight','normal',... 
       'FontSize',15);
saveas(gca, 'plot');
print('efficient_plot', '-dpdf', '-bestfit' ) ;
% close(h);

  % tex file
result_tab= horzcat( nuList, mean_test_pos_val', std_test_pos_val', ...
                 mean_test_neg_val', std_test_neg_val');
result_filename = 'value_tab.txt';
dlmwrite(result_filename, result_tab , '-append');

result_tex = 'value_tab.tex';
FID = fopen(result_tex, 'w');
fprintf(FID, '\\begin{tabular}{rrrrr}\\hline \n');

fprintf(FID, '$\\nu$  & $\\widehat{V}^{+}$ & $std^{+}$ & $\\widehat{V}^{-}$ & $std^{-}$ \\\\\\hline \n');
printtab = result_tab;
  for k=1:size(printtab,1)
      printline = printtab(k, :);
      fprintf(FID, '%8.2f & %8.2f & %8.2f & %8.2f  & %8.2f \\\\ ', printline);
      if k==size(printtab,1)
          fprintf(FID, '\\hline ');
      end
      fprintf(FID, '\n');
  end
  fprintf(FID, '\\end{tabular}\n');
  fclose(FID);
    
%quit force