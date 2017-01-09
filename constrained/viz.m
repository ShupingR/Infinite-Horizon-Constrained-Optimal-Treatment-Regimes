%% load data
load output_3.txt
dat = sort(output_3, 1);
dat = array2table(dat);
dat.Properties.VariableNames = {'k' 'kappa' 'fval' 'exitflag' 'tau1' 'tau2' 'tau3' 'tau4' 'tau5' 'tau6'};
dat = dat(5:height(dat),:);
tauTab = [ dat.tau1, dat.tau2, dat.tau3, dat.tau4, dat.tau5, dat.tau6 ];
kappaList = dat.kappa;
%% generate test dataset, do not retrain weights using test dataset!
N = 1000;
T = 7;
seed_train = 111;
seed_test = 222;
train_sample = sample_collect(N, T, seed_train);
test_sample = test_sample_collect(3*N, T, seed_test); 

%% plot pareto efficient frontier on training dataset
% calculate neg valfun
% train_val_neg = nan(length(kappaList),1);
% 
% for k = 1:length(kappaList)
%     kappa = kappaList(k);
%     tau = tauTab(k, :)';
%     train_val_neg(k) = constraint_function_two( tau, train_sample, kappa)  + kappa;
% end
% 
% h = figure;
% plot(kappaList, dat.fval, 'b--o', ...
%       kappaList, train_val_neg, 'r--*');
% title('Pareto Frontier on Training Dataset')
% xlabel('constraint upper bound kappa')
% ylabel('Value functions for positive rewards (blue) and negative rewards (red)')
% savefig(h, 'Pareto Frontier on Trainset.fig');
% close(h);


%% calculate weights and plot

% algorithm = 1;  
% maxiterations = 100;
% epsilon = 0.001;
% initial_policy_weights = ones(5*5 , 1);

%% plot q(s,a), v(s)
% [pos_state, neg_state] = meshgrid(-1.5: 0.1 :4.0, -1.5: 0.1 :4.0);
% pos_qsa_mat = nan(size(pos_state, 1),size(pos_state, 2), 5, length(kappaList)); 
% neg_qsa_mat = nan(size(pos_state, 1),size(pos_state, 2), 5, length(kappaList)); 
% pos_vs_pol_mat = nan(size(pos_state, 1),size(pos_state, 2), length(kappaList)); 
% neg_vs_pol_mat = nan(size(pos_state, 1),size(pos_state, 2), length(kappaList)); 
% act_pol_mat = nan(size(pos_state, 1),size(pos_state, 2), length(kappaList)); 
% test_val_pos = nan(length(kappaList), 1); 
% test_val_neg = nan(length(kappaList), 1);
% pos_weight_mat = nan(length(kappaList), 25);
% neg_weight_mat = nan(length(kappaList), 25);
% for k = 1: length(kappaList) 
%     tau = tauTab(k, :)';% column vector
%     [pos_weights, ~] = lspi_two_pos( algorithm, maxiterations, epsilon, ...
%                                                  train_sample, initial_policy_weights, tau);
%                                              
%     [neg_weights, ~] = lspi_two_neg( algorithm, maxiterations, epsilon, ...
%                                                   train_sample, initial_policy_weights, tau);
%                                               
%     [test_val_pos(k), test_val_neg(k) ] = ...
%         val_testset(tau, pos_weights, neg_weights, test_sample );
%     
%     pos_weight_mat(k, :) = pos_weights;
%     neg_weight_mat(k, :) = neg_weights;
% %     for i = 1: size(pos_state, 1)
% %         for j = 1: size(pos_state, 2)
% %             pos_s = pos_state(i, j);
% %             pos_phi = basis_rbf(pos_s, 1) + basis_rbf(pos_s, 2) + ...
% %                           basis_rbf(pos_s, 3) + basis_rbf(pos_s, 4) + ...
% %                           basis_rbf(pos_s, 5);
% %             
% %             pos_qsa = sum( vec2mat(pos_phi .* pos_weights, 5), 2);
% %             pos_qsa_mat(i, j, :, k) = pos_qsa';
% %             
% %             neg_s = neg_state(i, j);
% %             neg_phi = basis_rbf(neg_s, 1) + basis_rbf(neg_s, 2) + ...
% %                           basis_rbf(neg_s, 3) + basis_rbf(neg_s, 4) + ...
% %                           basis_rbf(neg_s, 5);
% %             
% %             neg_qsa = sum( vec2mat(neg_phi .* neg_weights, 5), 2);
% %             neg_qsa_mat(i, j, :, k) = neg_qsa';
% %         
% %             [ act_pol, actionphi_pos, actionphi_neg]= policy_function_two(tau, pos_s, neg_s);
% %             act_pol_mat(i, j, k) =  act_pol;
% %             pos_vs_pol_mat(i, j, k) = sum( actionphi_pos .* pos_weights ); %#o
% %             neg_vs_pol_mat(i, j, k) = sum( actionphi_neg .* neg_weights );
% %         end
% %     end
% %     
% %     h(1) = figure;
% %     h1= surface(pos_state, neg_state, pos_qsa_mat(:, :, 1, k));
% %     h2 = surface(pos_state, neg_state, pos_qsa_mat(:, :, 2, k));
% %     h3 = surface(pos_state, neg_state, pos_qsa_mat(:, :, 3, k));
% %     h4 = surface(pos_state, neg_state, pos_qsa_mat(:, :, 4, k));
% %     h5 = surface(pos_state, neg_state, pos_qsa_mat(:, :, 5, k));
% %     legend([h1, h2, h3, h4, h5], {'a1', 'a2', 'a3', 'a4', 'a5'});
% %     title( strcat('Q function for positive rewards, Q+(s,a) with constraint upperboud kappa = ' , ...
% %            num2str(k)) );
% %     xlabel('state: tumor size');
% %     ylabel('state: toxicity');   
% %     zlabel('Q function for positive rewards');
% %        
% %     h(2) = figure;
% %     h1 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 1, k));
% %     h2 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 2, k));
% %     h3 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 3, k));
% %     h4 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 4, k));
% %     h5 = surface(pos_state, neg_state, neg_qsa_mat(:, :, 5, k));
% %     legend([h1, h2, h3, h4, h5], {'a1', 'a2', 'a3', 'a4', 'a5'});
% %     title( strcat('Q function for positive rewards, Q-(s,a) with constraint upperboud kappa = ' , ...
% %            num2str(k)) );
% %     xlabel('state: tumor size');
% %     ylabel('state: toxicity');   
% %     zlabel('Q function for negative rewards');
% %        
% %   %  subplot( 7, 2, 11);
% %     h(3) = figure;
% %     subplot( 2, 1, 1);
% %     surface(pos_state, neg_state, pos_vs_pol_mat(:, :, k));
% %     title( strcat('Value function for positive rewards, V+(s,a) with constraint upperboud kappa = ' , ...
% %            num2str(k)) );
% %     xlabel('state: tumor size');
% %     ylabel('state: toxicity');   
% %     zlabel('Value function for positive rewards');
% %     subplot( 2, 1, 2);    
% %     surface(pos_state, neg_state, neg_vs_pol_mat(:, :, k));
% %     title( strcat('Value function for negative rewards, V-(s,a) with constraint upperboud kappa = ' , ...
% %            num2str(k)) );
% %     xlabel('state: tumor size');
% %     ylabel('state: toxicity');   
% %     zlabel('Value function for negative rewards');
% %     
% %     h(4) = figure;
% %     surface(pos_state, neg_state, act_pol_mat(:, :, k));
% %     title('policy action)');
% %     
% %     savefig(h,strcat('kappa', num2str(k), '.fig'));
% %     close(h)
% 
%     dlmwrite('pos_weights_3.txt', pos_weights', '-append', 'delimiter', '\t')
%     dlmwrite('neg_weights_3.txt', neg_weights', '-append', 'delimiter', '\t')
%     %fileID = fopen('neg_weights_3.txt','a');
%     %fprintf(fileID,'%4.4f\t',neg_weights);
% end   
% 


%% plot pareto efficient frontier on testing dataset, 

load pos_weights_3.txt;
load neg_weights_3.txt

test_val_pos = nan(length(kappaList), 1); 
test_val_neg = nan(length(kappaList), 1);
for k = 1: length(kappaList) 
    disp(k);
    tau = tauTab(k,:)';
    neg_weights = neg_weights_3(k,:)';
    pos_weights = pos_weights_3(k,:)';
    [test_val_pos(k), test_val_neg(k) ] = ...
        val_testset(tau, pos_weights, neg_weights, test_sample );
end
h = figure;
plot(kappaList, test_val_pos, 'b--o', ...
       kappaList, test_val_neg, 'r-*');
title('Pareto Frontier on Test Dataset')
xlabel('constraint upper bound kappa')
ylabel('Value functions for positive rewards (blue) and negative rewards (red)');
savefig(h, 'Pareto Frontier on Testset.fig');
close(h);

% %%  train states 3D histogram
% h = figure;
% hist3( [ vertcat(train_sample.state_pos), vertcat(train_sample.state_neg) ], [50 50]);
% xlabel('postive state: tumor size'); ylabel('negative state: toxicity');
% set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
% savefig(h, 'Histogram of states on trainset.fig');
% close(h);
% 
% %%  test states 3D histogram
% h = figure;
% hist3( [ vertcat(test_sample.state_pos), vertcat(test_sample.state_neg) ], [50 50]);
% xlabel('postive state: tumor size'); ylabel('negative state: toxicity');
% set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
% savefig(h, 'Histogram of states on testset.fig');
% close(h);