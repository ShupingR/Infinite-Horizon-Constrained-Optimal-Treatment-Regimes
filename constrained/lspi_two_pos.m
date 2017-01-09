function [policy_weights, all_policies] = lspi_two_pos(algorithm, maxiterations, epsilon, samples, initial_policy_weights, tau)
  
  %%% Initialize policy iteration 
  iteration = 0;
  distance = inf;
  all_policies{1} = initial_policy_weights;
  policy_weights = initial_policy_weights;
  
  %%% If no samples, return
  if isempty(samples)
    disp('Warning: Empty sample set');
    return
  end
  
  
  %%% Main LSPI loop  
  while ( (iteration < maxiterations) && (distance > epsilon) )
     disp('*****************************');
     disp(tau');
    %%% Update and print the number of iterations
    iteration = iteration + 1;
   % disp('*********************************************************');
   % disp( ['LSPI iteration : ', num2str(iteration)] );
    if (iteration==1)
      firsttime = 1;
    else
      firsttime = 0;
    end
    
 
    %%% You can optionally make a call to collect_samples right here
    %%% to change/update the sample set. Make sure firsttime is set
    %%% to 1 if you do so.
    

    %%% Evaluate the current policy (and implicitly improve)
    %%% There are several options here - choose one
    if (algorithm == 1)
      policy_weights = lsq_two_pos(samples, tau);
    elseif (algorithm == 2)
      policy_weights = lsqfast(samples, all_policies{iteration}, ...
			       policy_weights, firsttime);
    elseif (algorithm == 3)
      policy_weights = lsqbe(samples, all_policies{iteration}, policy_weights);
    elseif (algorithm == 4)
      policy_weights = lsqbefast(samples, all_policies{iteration}, ...
				 policy_weights, firsttime);
    end

    
    %%% Compute the distance between the current and the previous policy
    l1 = length(policy_weights);
    l2 = length(all_policies{iteration});
    if (l1 == l2)
      difference = policy_weights - all_policies{iteration};
      LMAXnorm = norm(difference,inf);
      L2norm = norm(difference);
    else
      LMAXnorm = abs(norm(policy_weights,inf) - ...
		     norm(all_policies{iteration},inf));
      L2norm = abs(norm(policy_weights) - ...
		   norm(all_policies{iteration}));
    end
    distance = L2norm;
      
      
    
    %%% Print some information 
    %disp( ['   Norms -> Lmax : ', num2str(LMAXnorm), ...
	 %  '   L2 : ',            num2str(L2norm)] );
    
    
    %%% Store the current policy
    all_policies{iteration+1} = policy_weights; %#ok<AGROW>
    
    
    %%% Depending on the domain, print additional info if needed
    % feval('my_print_info', all_policies);
    
  end
  
  
  %%% Display some info
 %%%%% disp('*********************************************************');
  if (distance > epsilon) 
    disp(['LSPI finished in ' num2str(iteration) ...
	  ' iterations WITHOUT CONVERGENCE to a fixed point']);
  else
    disp(['LSPI converged in ' num2str(iteration) ' iterations']);
  end
  disp('********************************************************* ');
  
  
end
