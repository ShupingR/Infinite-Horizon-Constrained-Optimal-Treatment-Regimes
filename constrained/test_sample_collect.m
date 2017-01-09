% generate sample with neg reward and positive reward together
% normalized and scaled W and M
% test everyone at the beginning

function sample = test_sample_collect(N, T, seed)
    % sample size N
    % stage number T
    % k feature dimension for each action
    rng(seed,'twister');
    % Dt, action variable, the chemotherapy agent dose level 
    % Wt, state variable, measures the negative part of wellness (toxicity)
    % Mt, state variable, denotes the tumor size 
    % all at decision time t
    % Create Matrix of W M D
    W = nan(N, T);
    M = nan(N, T);

    
    % The initial values W1 (W0) and M1 (M0) for each patient are generated 
    % iid from uniform (0, 2). 
    W(:,1) = 0 + (2 - 0) .* rand(N,1);
    M(:,1) = 0 + (2 - 0) .* rand(N,1); 
    
    mW =  1.5286;
    sW = 0.9230;
    mM = 1.4731;
    sM = 1.0121;
 
    % standardize input
    W = ( W - mW ) / sW;
    M = ( M - mM ) / sM ;
    
    field1 = 'state_pos';  value1 = nan;
    field2 = 'state_neg';  value2 = nan;
    field3 = 'absorb'; value3 = nan;
    field4 = 'action';  value4 = nan;
    field5 = 'reward_pos';  value5 = nan;
    field6 = 'reward_neg';  value6 = nan;
    field7 = 'nextstate_pos';  value7 = nan;
    field8 = 'nextstate_neg';  value8 = nan;
    sample= struct(field1,value1,field2,value2,field3,value3,field4,value4, ...
                         field5,value5, field6,value6, field7,value7, field8,value8);
   

    for i = 1:N
        % all the positive reward samples
        sample(i).state_pos= M(i);
        sample(i).state_neg = W(i);
        sample(i).absorb = nan;
        sample(i).action =  nan;
        sample(i).reward_pos = nan;
        sample(i).reward_neg = nan;
        sample(i).nextstate_pos = nan;
        sample(i).nextstate_neg = nan;            
    end
end