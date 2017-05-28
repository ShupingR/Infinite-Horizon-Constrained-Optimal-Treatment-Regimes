%%----------------------------------------------------------------%%
%  plot for showing treatment assignement for each state.    %
%%----------------------------------------------------------------%%

%% load data
% test samples
K = 4; % number of radial basis functions, not include the intercept
L = 5; % number of dosage levels
N = 7000; % training set sample size
T = 7; % number of stages
discount = 0.8; 
test_seed = 1001;
rng(test_seed,'twister');
test_sample = sample_collect(N, T, K, test_seed); % generate training set

prctile_state_pos = [ -0.9539, -0.3130, 0.1992, 0.8149 ];
dist_state_pos = [ 1.0951, 0.8306,  0.8297, 1.0882 ];
prctile_state_neg = [   -0.9003,  -0.3267,  0.1864, 0.8195 ];
dist_state_neg = [ 1.0588, 0.8271, 0.8248, 1.0903 ];

% constrained par
cd ~/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/results/may_24/constrained/
rep = 1;
fileName = [ 'output_may_16_constrained_sequential_initial_rep_' num2str(rep) ]; 
dataStruct.(fileName) =  load( [ fileName '.txt' ] );
dat = sortrows(dataStruct.(fileName) , 1); % sort the result by first column
dat = array2table(dat);
dat.Properties.VariableNames = {'k' 'nu' 'objective_val' 'constraint_val' 'exitflag' ...
                                              'tau0' 'tau1' 'tau2' 'tau3' 'tau4' 'tau5'};
tauTab = [ dat.tau0, dat.tau1, dat.tau2, dat.tau3, dat.tau4, dat.tau5 ];
nuList = dat.nu; % constraint value on secondary value 

%%             
cd ~/GitHub/research-github/Infinite-Horizon-Constrained-Optimal-Treatment-Regimes/plots
% plot q(s,a), v(s)
[pos_state, neg_state] = meshgrid(-1.5: 0.1 :4.0, -1.5: 0.1 :4.0);
% weights for q functions 
pos_weight_mat = nan(length(nuList), 25);
neg_weight_mat = nan(length(nuList), 25);
    % q function value for each state and action
pos_qsa_mat = nan(size(pos_state, 1), size(pos_state, 2), 5, length(nuList)); 
neg_qsa_mat = nan(size(neg_state, 1), size(neg_state, 2), 5, length(nuList)); 
    % v function value for each state and action
pos_vs_pol_mat = nan(size(pos_state, 1), size(pos_state, 2), length(nuList)); 
neg_vs_pol_mat = nan(size(neg_state, 1), size(neg_state, 2), length(nuList)); 
    % action for each state
act_pol_mat = nan(size(pos_state, 1), size(pos_state, 2), length(nuList)); 
%     test_val_pos = nan(length(nuList), 1); 
%     test_val_neg = nan(length(nuList), 1);

for k = 1:1
    tau = tauTab(k, :)';% column vector
    [pos_weights, ~, ~] = lsq_pos(tau, test_sample, discount, K, L); 
    [neg_weights, ~, ~] = lsq_neg(tau, test_sample, discount, K, L);                                          
    pos_weight_mat(k, :) = pos_weights;
    neg_weight_mat(k, :) = neg_weights;
    % loop through each state to calculate its q function and v function 
    for i = 1: size(pos_state, 1)
        for j = 1: size(pos_state, 2)
            pos_s = pos_state(i, j);
            pos_phi = basis_fast(pos_s, 1, prctile_state_pos, dist_state_pos) + ...
                          basis_fast(pos_s, 2, prctile_state_pos, dist_state_pos) + ...
                          basis_fast(pos_s, 3, prctile_state_pos, dist_state_pos) + ...
                          basis_fast(pos_s, 4, prctile_state_pos, dist_state_pos) + ...
                          basis_fast(pos_s, 5, prctile_state_pos, dist_state_pos);
            pos_qsa = sum( vec2mat(pos_phi .* pos_weights, 5), 2);
            pos_qsa_mat(i, j, :, k) = pos_qsa';

            neg_s = neg_state(i, j);
            neg_phi = basis_fast(neg_s, 1, prctile_state_neg, dist_state_neg) + ...
                          basis_fast(neg_s, 2, prctile_state_neg, dist_state_neg) + ...
                          basis_fast(neg_s, 3, prctile_state_neg, dist_state_neg) + ...
                          basis_fast(neg_s, 4, prctile_state_neg, dist_state_neg) + ...
                          basis_fast(neg_s, 5, prctile_state_neg, dist_state_neg);
            neg_qsa = sum( vec2mat(neg_phi .* neg_weights, 5), 2);
            neg_qsa_mat(i, j, :, k) = neg_qsa';

            neg_qsa = sum( vec2mat(neg_phi .* neg_weights, 5), 2);
            neg_qsa_mat(i, j, :, k) = neg_qsa';

            [ act_pol, actionphi_pos, actionphi_neg] = ...
                policy_function_deterministic_state(tau, pos_s, neg_s, ...
                    prctile_state_pos, dist_state_pos, prctile_state_neg, dist_state_neg);
            act_pol_mat(i, j, k) =  act_pol;
            pos_vs_pol_mat(i, j, k) = sum( actionphi_pos .* pos_weights ); %#o
            neg_vs_pol_mat(i, j, k) = sum( actionphi_neg .* neg_weights );
        end
    end
    
    % plot for Q+(s,a)
    h_1 = figure; 
    view(3)
    h1= surface(pos_state, neg_state, pos_qsa_mat(:, :, 1, k));
    h2 = surface(pos_state, neg_state, pos_qsa_mat(:, :, 2, k));
    h3 = surface(pos_state, neg_state, pos_qsa_mat(:, :, 3, k));
    h4 = surface(pos_state, neg_state, pos_qsa_mat(:, :, 4, k));
    h5 = surface(pos_state, neg_state, pos_qsa_mat(:, :, 5, k));
    legend([h1, h2, h3, h4, h5], {'a1', 'a2', 'a3', 'a4', 'a5'});
    title( {['$Q^{+}(s,a)$ with constraint upperboud $\nu$ = ' , ...
           num2str(nuList(k))]},  'interpreter' ,'latex', 'FontSize',15);
    xlabel({'State: tumor size'},  'interpreter' ,'latex', 'FontSize',15);
    ylabel({'State: toxicity'}, 'interpreter' ,'latex', 'FontSize',15);   
    zlabel({'Q function for positive rewards'},  'interpreter' ,'latex', 'FontSize',15);
    set(gca, 'Units','normalized', ...
         'FontUnits','points',... 
         'FontWeight','normal',... 
         'FontSize',15);
    print(strcat('q_pos_nu', num2str(k)), '-dpdf', '-bestfit' ); 
    close(h_1)
    
    % plot for Q-(s,a)
    h_2 = figure;
    view(3)
    h1 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 1, k));
    h2 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 2, k));
    h3 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 3, k));
    h4 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 4, k));
    h5 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 5, k));
    legend([h1, h2, h3, h4, h5], {'a1', 'a2', 'a3', 'a4', 'a5'});
    title( {['$Q^{-}(s,a)$ with constraint upperboud $\nu$ = ' , ...
           num2str(nuList(k))]},  'interpreter' ,'latex', 'FontSize',15);
    xlabel({'State: tumor size'},  'interpreter' ,'latex', 'FontSize',15);
    ylabel({'State: toxicity'}, 'interpreter' ,'latex', 'FontSize',15);   
    zlabel({'Q function for negative rewards'},  'interpreter' ,'latex', 'FontSize',15);
    set(gca, 'Units','normalized', ...
         'FontUnits','points',... 
         'FontWeight','normal',... 
         'FontSize',15);
    print(strcat('q_neg_nu', num2str(k)), '-dpdf', '-bestfit' ); 
    close(h_2)
    
   % plot for V+(s) 
    h_3_1 = figure;
    view(3)
    surface(pos_state, neg_state, pos_vs_pol_mat(:, :, k));
    title( {[ '$V^{+}(s)$ with constraint upperboud $\nu$ = ' , ...
           num2str(nuList(k))]}, 'interpreter' ,'latex', 'FontSize',15 );
    xlabel({'State: tumor size'}, 'interpreter' ,'latex', 'FontSize',15);
    ylabel({'State: toxicity'}, 'interpreter', 'latex', 'FontSize', 15);   
    zlabel({'Value function for positive rewards'},'interpreter', 'latex', 'FontSize', 15);
    set(gca, 'Units','normalized', ...
         'FontUnits','points',... 
         'FontWeight','normal',... 
         'FontSize',15);
    print(strcat('state_value_nu', num2str(k)), '-dpdf', '-bestfit' ); 
    close(h_3_1)
    
    % and V-(s)
    h_3_2 = figure;
    view(3)
    surface(pos_state, neg_state, neg_vs_pol_mat(:, :, k));
    title({['$\widehat{V}^{-}(s)$ of estimated constrained optimal regime, $\nu$ = ' , ...
           num2str(nuList(k))]}, 'interpreter' ,'latex', 'FontSize',15);
    xlabel({'State $M$: tumor size'}, 'interpreter' ,'latex', 'FontSize',15);
    ylabel({'State $W$: toxicity'}, 'interpreter', 'latex', 'FontSize', 15);   
    zlabel({'$\widehat{V}^{-}(s)$'},'interpreter', 'latex', 'FontSize', 15);
    set(gca, 'Units','normalized', ...
         'FontUnits','points',... 
         'FontWeight','normal',... 
         'FontSize',15);
    print(strcat('state_value_nu', num2str(k)), '-dpdf', '-bestfit' ); 
    close(h_3_2)
    
    % plot for action each state
     h_4= figure;
     view(2)
     surface(pos_state, neg_state, act_pol_mat(:, :, k));
     xlabel({'State variable $M$'}, 'interpreter' ,'latex', 'FontSize',15 )
     ylabel({'State variable $W$'}, 'interpreter' ,'latex', 'FontSize',15 )
     title({[ 'Action for each state under estimated' ; ...
              'constrained optimal regime, $\nu$ =' , ...
              num2str(nuList(k))]}, 'interpreter' ,'latex', 'FontSize',15);
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
    print(strcat('action_nu', num2str(k)), '-dpdf', '-bestfit' );     
    close(h_4)

    %dlmwrite('pos_weights_3.txt', pos_weights', '-append', 'delimiter', '\t')
    %dlmwrite('neg_weights_3.txt', neg_weights', '-append', 'delimiter', '\t')
    %fileID = fopen('neg_weights_3.txt','a');
    %fprintf(fileID,'%4.4f\t',neg_weights);
end
