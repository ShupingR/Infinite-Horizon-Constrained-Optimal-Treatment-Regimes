%----------------
% policy function, the minimal policy
%----------------
function action = policy(all_feature, tau)

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
%     
end
