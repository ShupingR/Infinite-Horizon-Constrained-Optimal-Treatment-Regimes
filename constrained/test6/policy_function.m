function [action, actionphi] = policy_function(policy_weights, state)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright 2000-2002
%
% Michail G. Lagoudakis (mgl@cs.duke.edu)
% Ronald Parr (parr@cs.duke.edu)
%
% Department of Computer Science
% Box 90129
% Duke University, NC 27708
%
%
% [action, actionphi] = policy_function(policy, state)
%
% Computes the "policy" at the given "state".
%
% Returns the "action" that the policy picks at that "state" and the
% evaluation ("actionphi") of the basis at the pair (state, action).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%% Pick the action with maximum Q-value
    bestq = -inf;
    besta = [];
    %%% Find first all actions with maximum Q-value
    for i = 1:5
      phi = basis_rbf(state, i);
      q = basis_rbf(state, i)' * policy_weights;
       
      if (q > bestq)
    	bestq = q;
        besta = [i]; %#ok<*NBRAK>
        actionphi = [phi];
      elseif (q == bestq)
        besta = [besta; i]; %#ok<*AGROW>
        actionphi = [actionphi, phi];
      end

    end

    %%% Now, pick one of them
    which = 1;                         % Pick the first (deterministic)
    %which = randint(length(besta));    % Pick randomly

    action = besta(which);
    actionphi = actionphi(:,which);

end


