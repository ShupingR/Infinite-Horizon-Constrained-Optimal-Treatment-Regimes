function [w, A, b] = lsq_pos(tau, sample, discount, K, L)
%------------------------------------------------------------------------------------------------------
% Reference: LSPI
% LSQ for policy evaluation in terms of negative reward, aka,
% Q function approximation for negative reward under a fixed policy
%
% Evaluates the "policy" using the set of "sample", that is, it
% learns a set of weights for the basis specified to
% form the approximate Q-value of the "policy".
% The approximation is the fixed point of the Bellman equation.
%
% Returns the learned weights w and the matrices A and b of the
% linear system Aw=b. 
%
% Input : 
% sample: sample with the structure specified
% tau: policy index
% discount: discount factor
% K: number of radial basis functions
% L: number of dosage levels
% Output:
% w: weight for Q function linear approximation in terms of negative reward
% A: linear system matrix A
% b: linear system vector b
%------------------------------------------------------------------------------------------------------
  
  howmany = length(sample);
  p = (K + 1)*L;
  A = zeros(p, p);
  b = zeros(p, 1);
  % mytime = cputime;

  %%% Loop through the sample 
  for i=1:howmany
    %%% Compute the basis for the current state and action
   % phi = basis(sample(i).state_pos, sample(i).action, sample(i).prctile_state_pos, ...
   %        sample(i).dist_state_pos, K, L);
    phi = basis_fast(sample(i).state_pos, sample(i).action, sample(i).prctile_state_pos, ...
            sample(i).dist_state_pos);
    %%% Make sure the nextstate is not an absorbing state
    if ~sample(i).absorb
      %%% Compute the policy and the corresponding basis at the next state 
        nextaction = policy_function_deterministic(tau, sample(i));
        nextphi = basis_fast(sample(i).nextstate_pos, nextaction, sample(i).prctile_state_pos, ...
                      sample(i).dist_state_pos);
    else
        nextphi = zeros(p, 1);
    end
    % check values of phi and nextphi
    
    
    %%% Update the matrices A and b
    A = A + phi * (phi - discount * nextphi)';
    b = b + phi * sample(i).reward_pos;
%    phi_mat(i, :) = phi';
 %   nextphi_mat(i, :) = nextphi';
  end

  %phi_time = cputime - mytime;
  %disp(['CPU time to form A and b : ' num2str(phi_time)]);
  %mytime = cputime;
  
  %%% Solve the system to find w
  rankA = rank(A);
  
  %rank_time = cputime - mytime;
  %disp(['CPU time to find the rank of A : ' num2str(rank_time)]);
  %mytime = cputime;
  
  %disp(['Rank of matrix A : ' num2str(rankA)]);
  if rankA==p
    %disp('A is a full rank matrix!!!');
    w = A\b;
  else
    %disp(['WARNING: A is lower rank!!! Should be ' num2str(k)]);
    w = pinv(A)*b;
  end
  
 % solve_time = cputime - mytime;
 % disp(['CPU time to solve Aw=b : ' num2str(solve_time)]);
  
  
end
  
