%----------------
% policy function, the minimal policy
%----------------
function best_d = policy(m, w, f, med_m, med_w, weights,k)

    %%% Pick the action with maximum Q-value
    fun = @(d) qfun(m, w, d, f, med_m, med_w, weights,k);
    %options = optimoptions(@fminunc,'Algorithm', 'quasi-newton', ...
    %                       'Display','off' , 'FinDiffRelStep', 1e-2);
    %problem = createOptimProblem('fminunc', 'objective', fun, 'x0', 0.5, ...
    %                              'options', options);
    %ms = MultiStart('StartPointsToRun', 'all', 'Display','off');
    %[best_d] = run(ms, problem, 5);
    % d_starts = [0.1, 0.3, 0.6, 0.9];
    % min_qval = 1000;
    %for d_start = d_starts
    % [d_sol, qval_sol] = fminbnd(fun, 0, 1);
        % if (qval_sol < min_qval)
    best_d = fminbnd(fun, 0, 1);
    % best_d = d_sol;
           % min_qval = qval_sol;
    %    else
    %        ...
    %    end
    %end

end
