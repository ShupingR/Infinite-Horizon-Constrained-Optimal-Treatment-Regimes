%% plot for showing treatment assignement for any state.

%% load data

REP = 2;
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
    cd ~/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/results/may_24/constrained/
    fileName = [ 'output_may_16_constrained_sequential_initial_rep_' num2str(rep) ]; 
    dataStruct.(fileName) =  load( [ fileName '.txt' ]);
    dat = sortrows(dataStruct.(fileName) , 1); % sort the result by first column
    dat = array2table(dat);
    dat.Properties.VariableNames = {'k' 'nu' 'objective_val' 'constraint_val' 'exitflag' ...
                                                  'tau0' 'tau1' 'tau2' 'tau3' 'tau4' 'tau5'};
    tauTab = [ dat.tau0, dat.tau1, dat.tau2, dat.tau3, dat.tau4, dat.tau5 ];
    nuList = dat.nu; % constraint value on secondary value 
    
    %% generate train dataset 
    cd ~/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/
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
    cd ~/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/results/may_24/unconstrained/
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
             
cd ~/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/results/

% plot q(s,a), v(s)
[pos_state, neg_state] = meshgrid(-1.5: 0.1 :4.0, -1.5: 0.1 :4.0);
pos_qsa_mat = nan(size(pos_state, 1),size(pos_state, 2), 5, length(nuList)); 
neg_qsa_mat = nan(size(pos_state, 1),size(pos_state, 2), 5, length(nuList)); 
pos_vs_pol_mat = nan(size(pos_state, 1),size(pos_state, 2), length(nuList)); 
neg_vs_pol_mat = nan(size(pos_state, 1),size(pos_state, 2), length(nuList)); 
act_pol_mat = nan(size(pos_state, 1),size(pos_state, 2), length(nuList)); 
test_val_pos = nan(length(nuList), 1); 
test_val_neg = nan(length(nuList), 1);
pos_weight_mat = nan(length(nuList), 25);
neg_weight_mat = nan(length(nuList), 25);
for k = 15
    tau = tauTab(k, :)';% column vector
    [pos_weights, ~] = lspi_two_pos( algorithm, maxiterations, epsilon, ...
                                                 test_sample, initial_policy_weights, tau );
                                             
    [neg_weights, ~] = lspi_two_neg( algorithm, maxiterations, epsilon, ...
                                                  test_sample, initial_policy_weights, tau );
                                              
    [test_val_pos(k), test_val_neg(k) ] = ...
        val_testset(tau, pos_weights, neg_weights, test_sample );
    
    pos_weight_mat(k, :) = pos_weights;
    neg_weight_mat(k, :) = neg_weights;
    for i = 1: size(pos_state, 1)
        for j = 1: size(pos_state, 2)
            pos_s = pos_state(i, j);
            pos_phi = basis_rbf(pos_s, 1) + basis_rbf(pos_s, 2) + ...
                          basis_rbf(pos_s, 3) + basis_rbf(pos_s, 4) + ...
                          basis_rbf(pos_s, 5);
            
            pos_qsa = sum( vec2mat(pos_phi .* pos_weights, 5), 2);
            pos_qsa_mat(i, j, :, k) = pos_qsa';
            
            neg_s = neg_state(i, j);
            neg_phi = basis_rbf(neg_s, 1) + basis_rbf(neg_s, 2) + ...
                          basis_rbf(neg_s, 3) + basis_rbf(neg_s, 4) + ...
                          basis_rbf(neg_s, 5);
            
            neg_qsa = sum( vec2mat(neg_phi .* neg_weights, 5), 2);
            neg_qsa_mat(i, j, :, k) = neg_qsa';
        
            [ act_pol, actionphi_pos, actionphi_neg]= policy_function_two(tau, pos_s, neg_s);
            act_pol_mat(i, j, k) =  act_pol;
            pos_vs_pol_mat(i, j, k) = sum( actionphi_pos .* pos_weights ); %#o
            neg_vs_pol_mat(i, j, k) = sum( actionphi_neg .* neg_weights );
        end
    end
    
    h(1) = figure;
    h1= surface(pos_state, neg_state, pos_qsa_mat(:, :, 1, k));
    h2 = surface(pos_state, neg_state, pos_qsa_mat(:, :, 2, k));
    h3 = surface(pos_state, neg_state, pos_qsa_mat(:, :, 3, k));
    h4 = surface(pos_state, neg_state, pos_qsa_mat(:, :, 4, k));
    h5 = surface(pos_state, neg_state, pos_qsa_mat(:, :, 5, k));
    legend([h1, h2, h3, h4, h5], {'a1', 'a2', 'a3', 'a4', 'a5'});
    title( strcat('Q function for positive rewards, Q+(s,a) with constraint upperboud kappa = ' , ...
           num2str(k)) );
    xlabel('state: tumor size');
    ylabel('state: toxicity');   
    zlabel('Q function for positive rewards');
       
    h(2) = figure;
    h1 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 1, k));
    h2 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 2, k));
    h3 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 3, k));
    h4 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 4, k));
    h5 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 5, k));
    legend([h1, h2, h3, h4, h5], {'a1', 'a2', 'a3', 'a4', 'a5'});
    title( strcat('Q function for positive rewards, Q-(s,a) with constraint upperboud kappa = ' , ...
           num2str(k)) );
    xlabel('state: tumor size');
    ylabel('state: toxicity');   
    zlabel('Q function for negative rewards');
       
  %  subplot( 7, 2, 11);
    h(3) = figure;
    subplot( 2, 1, 1);
    surface(pos_state, neg_state, pos_vs_pol_mat(:, :, k));
    title( strcat('Value function for positive rewards, V+(s,a) with constraint upperboud kappa = ' , ...
           num2str(k)) );
    xlabel('state: tumor size');
    ylabel('state: toxicity');   
    zlabel('Value function for positive rewards');
    subplot( 2, 1, 2);    
    surface(pos_state, neg_state, neg_vs_pol_mat(:, :, k));
    title( strcat('Value function for negative rewards, V-(s,a) with constraint upperboud kappa = ' , ...
           num2str(k)) );
    xlabel('state: tumor size');
    ylabel('state: toxicity');   
    zlabel('Value function for negative rewards');
    
     h = figure;
     surface(pos_state, neg_state, act_pol_mat(:, :, k));
   
     xlabel({'State variable $M$'}, ...
          'interpreter' ,'latex', 'FontSize',15 )
     ylabel({'State variable $W$'},...
          'interpreter' ,'latex', 'FontSize',15 )
     title({'Action for each state under the estimated constrained optimal regime'},...
          'interpreter' ,'latex', 'FontSize',15);
  %   legend({'$\widehat{V}^{+}$ vs. $\nu$',  '$\widehat{V}^{-}$ vs. $\nu$'}, ...
   %        'interpreter' ,'latex', 'Location','SouthEast','FontSize',15);
  % axis([0 t(end) -1.5 1.5]);
    c = colorbar('Ticks',[1,2,3,4,5],...
         'TickLabels',{'0.00','0.25','0.50','0.75','1.0'});
    c.Label.String = 'Action (Dosage)';
     set(gca, 'Units','normalized', ...
         'FontUnits','points',... 
         'FontWeight','normal',... 
         'FontSize',15);
     print(strcat('kappa', num2str(k)), '-dpdf', '-bestfit' );     
     close(h);

    dlmwrite('pos_weights_3.txt', pos_weights', '-append', 'delimiter', '\t')
    dlmwrite('neg_weights_3.txt', neg_weights', '-append', 'delimiter', '\t')
    %fileID = fopen('neg_weights_3.txt','a');
    %fprintf(fileID,'%4.4f\t',neg_weights);
end   




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
xlabel({'$\nu$ bounds on secondary potential outcome'}, 'interpreter' ,'latex', 'FontSize',15 )
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

