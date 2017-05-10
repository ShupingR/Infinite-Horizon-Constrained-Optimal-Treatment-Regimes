clear;
%profile on; 
% Training to get q function's weight and policy index
nk = 40; % number of bounds
%npar = 40; % parallel number 
ns = 10; % number of random start
K = 4; % number of radial basis functions, not include the intercept
L = 5; % number of dosage levels
N = 6000; % training set sample size
T = 7; % number of stages
discount = 0.8;
seed = 111;
rng(seed,'twister');
sample = sample_collect(N, T, K, seed); % generate training set
fileName = 'tau_val.txt';
sign_min = 1; % to minimize a function
sign_max = -1; % to maximize a function
pos_reward = 1; % calculate wrt positive reward
neg_reward = -1; % calculate wrt negative reward
O = nan;
C = nan;
tau1List = -10:0.5:10;
tau2List = -10:0.5:10;
[tau1, tau2] = meshgrid(tau1List, tau2List);
npar = 24;
parpool(npar)  

for i = 1:size(tau1, 1) 
    parfor j = 1:size(tau1, 2); 
        disp(i);
        tau11 = tau1(i,j);
        tau22 = tau2(i,j);
        tau = [ tau11; tau22 ];
        O(i, j)  = value_function( tau, sample, discount, K, L, pos_reward, sign_min );
        C(i, j) = value_function( tau, sample, discount, K, L, neg_reward, sign_min );
        disp(j);
    end
end

oFig = figure;
surf(tau1, tau2, O);
saveas(gcf,'obj.fig');
close(oFig)

cFig = figure;
surf(tau1, tau2, C);
saveas(gcf,'const.fig');
close(cFig)

fclose('all');
%profile viewer;
%profsave;
%profile off;
%quit force;