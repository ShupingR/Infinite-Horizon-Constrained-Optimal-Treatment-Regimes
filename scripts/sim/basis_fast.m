%----------------------------------------------------------------------------%
% construct the basis functions for the action taken: radial basis function  %
% computes a number of radial basis functions (on state) with means equal to %
% percentiles. This block of basis functions is duplicated for each action.  %
% The "action" determines which segment will be active.                      %
%----------------------------------------------------------------------------%

% checkmark
function phi = basis_fast(state, action, prctile_state, dist_state)
% input:
% state : state variable, a scalar
% action: action taken
% K = 4 not including intercept
% L = 5 dose level
% output:
% a vector of basis function

%%% Initialize  
    phi = zeros(25,1); % phi = nan((K+1)*L,1);

%%% Compute the RBFs
    if action == 1
        phi(1) = 1;
        phi(2) = exp( - (state - prctile_state(1))^2 / dist_state(1)^2 );
        phi(3) = exp( - (state - prctile_state(2))^2 / dist_state(2)^2 );
        phi(4) = exp( - (state - prctile_state(3))^2 / dist_state(3)^2 );
        phi(5) = exp( - (state - prctile_state(4))^2 / dist_state(4)^2 );
    elseif action == 2
        phi(6) = 1;
        phi(7) = exp( - (state - prctile_state(1))^2 / dist_state(1)^2 );
        phi(8) = exp( - (state - prctile_state(2))^2 / dist_state(2)^2 );
        phi(9) = exp( - (state - prctile_state(3))^2 / dist_state(3)^2 );
        phi(10) = exp( - (state - prctile_state(4))^2 / dist_state(4)^2 );
    elseif action == 3
        phi(11) = 1;
        phi(12) = exp( - (state - prctile_state(1))^2 / dist_state(1)^2 );
        phi(13) = exp( - (state - prctile_state(2))^2 / dist_state(2)^2 );
        phi(14) = exp( - (state - prctile_state(3))^2 / dist_state(3)^2 );
        phi(15) = exp( - (state - prctile_state(4))^2 / dist_state(4)^2 );
    elseif action == 4
        phi(16) = 1;
        phi(17) = exp( - (state - prctile_state(1))^2 / dist_state(1)^2 );
        phi(18) = exp( - (state - prctile_state(2))^2 / dist_state(2)^2 );
        phi(19) = exp( - (state - prctile_state(3))^2 / dist_state(3)^2 );
        phi(20) = exp( - (state - prctile_state(4))^2 / dist_state(4)^2 );
    elseif action == 5
        phi(21) = 1;
        phi(22) = exp( - (state - prctile_state(1))^2 / dist_state(1)^2 );
        phi(23) = exp( - (state - prctile_state(2))^2 / dist_state(2)^2 );
        phi(24) = exp( - (state - prctile_state(3))^2 / dist_state(3)^2 );
        phi(25) = exp( - (state - prctile_state(4))^2 / dist_state(4)^2 );
    else
        disp('wrong action number')
    end
end


