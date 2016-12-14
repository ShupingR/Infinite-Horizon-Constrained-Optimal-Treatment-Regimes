function phi = basis_rbf(state, action)

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
% phi = chain_basis_rbf(state, action)
%
% Computes a number of radial basis functions (on "state") with means
% spread uniformly over the chain. This block of basis functions is
% duplicated for each action. The "action" determines which segment
% will be active.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  A = 5;
  nrbf = 4;
%  SS = 2*S/nrbf;
%   action = (action == 0) * 1 + ...
%               (action == 0.25) * 2 + ...
%               (action == 0.5) * 3 + ...
%               (action == 0.75) * 4 + ...
%               (action == 1) * 5;
  %%% The RBFs and a constant is repeated for each action
  numbasis = (nrbf+1)*A;
  
  %%% Initialize
  phi = zeros(numbasis,1);
  %%% Find the starting position
  base = (action-1) * (nrbf+1);
  
  %%% Compute the RBFs
  cent = [ -0.9393 ; -0.4044 ; 0.0719 ; 0.6418 ];
  dist = [ 0.9728 ; 0.7615 ; 0.7611; 1.0004]; 
 % disp('action');
 % disp(action);
  for i=1:nrbf
   %disp('i');
    %disp(i);
    phi(base+1+i) = exp(-norm(state-cent(i))^2/dist(i)^2);
  end
  
  %%% ... and the constant!
  phi(base+1) = 1;
  
  return
  
