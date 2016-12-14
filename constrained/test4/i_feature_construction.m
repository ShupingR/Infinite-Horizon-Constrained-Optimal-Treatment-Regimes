%-------------------------------------
% feature construction for policy functions
% input is a batch of sample
% return of the features of all actions
% the idea of radial basis function is to work on a higher dimensional space
% that is more likely to be seperable than the original space
%-------------------------------------
function phi = i_feature_construction (...
                    wellness, tumorsize, death, center_sig_mat, K )
    %-------------------------------------
    % feature construction for all Q(s,a) functions action and rewards
    % feature orders matters, as it correponds to which action and rewards
    % input sample a batch of single sample 
    % the number of Q functions  = |A|*|R| 
    % K is the number of features for each action 
    % TK is the number of total features for all actions and rewards 
    % K * |R| * |A|
    % assume all input are column vectors
    %-------------------------------------

    %-------------------------------------
    % feature construction for Q(s,a) functions
    % the number of Q functions  = |A|*|R| 
    % input sample is a single sample 
    % K is the number of features for each action 
    % TK is the number of total features for all actions and rewards 
    % K * |R| * |A|
    %------------------------------------- 
    
    % positive and negative rewards are using the same features,
    % fits seperately 
    
    % a single sample as input
    if ( death == 1 )            
        % absorb state
        phi = zeros(6, K); % which action            
    else
        % dimesion of feature increase from 2 to TK
        cube_each_state = nan(6, K, 2);
        each_state = [ wellness, tumorsize];
        each_state_1 = repmat(each_state(1), 1, K);
        each_state_2 = repmat(each_state(2), 1, K);
        cube_each_state( :, :, 1) = ...
                vertcat( each_state_1, each_state_1, ... 
                         each_state_1, each_state_1, ...
                         each_state_1, each_state_1 )...
                - center_sig_mat(:, :, 1);
        cube_each_state( :, :, 2) = ...
                vertcat( each_state_2, each_state_2, ... 
                         each_state_2, each_state_2, ...
                         each_state_2, each_state_2 ) ...
                - center_sig_mat(:, :, 2);
        inside_phi = ...
                (cube_each_state(:, :, 1).^2 ...
                + cube_each_state(:, :, 2).^2)...
                ./ (2 * center_sig_mat(:, :, 3).^2);
        phi = exp(-1 * inside_phi); % row vector
        
     end

end
   