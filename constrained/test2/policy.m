%----------------
% policy function
% softmax action selection
%----------------
function act_pol = policy(tau, phi)
%act_pol = policy(tau, weight, phi)
    % [~, act_pol] = max( phi * weight )
    %qval = phi * weight;
    %act_prob = exp(tau*qval) ./  (1 + exp(tau*qval));
    %[~, act_pol] = max(act_prob);
    
    % softmax 
    % calcualte all the qvalues
    % qval_pos = phi * weight_pos;
    % qval_neg = phi * weight_neg;
    % act_prob = exp(tau.*qval) ./  sum((1 + exp(tau.*qval)));
    %act_pol = randsample(1:6, 1, true, act_prob');
    % act_pol = 1;
   % [~, act_pol] = max(exp(phi * tau) ./  sum(exp(phi * tau)));
   % act_prob = exp(phi * tau) ./  sum(exp(phi * tau));
 
   act_prob = exp(phi * tau) ./  sum(exp(phi * tau));
   [~, act_pol] = max(act_prob);
    
    % optimal policy
   % [~, act_pol] = max( phi * weight );
  %  display(act_pol)
    % act_pol = randsample(1:6, 1, true, act_prob');
end

%     fw = all_feature * tau';
%     display('*****tau: policy*****');
%     display(tau);
%     display('*****fw: policy*****');
%     display(fw);
%     action = 0 * (fw < -0.5 ) + ...
%              0.2 * ( ( -0.5 < fw ) & (fw < -0.25) ) + ...
%              0.4 * ( (-0.25 < fw) & (fw < 0) ) + ...
%              0.6 * ( (0 < fw) & (fw < 0.25) ) + ...
%              0.8 * ( (0.25 < fw) & (fw < 0.5) ) + ...
%              1 * ( fw > 0.5 );