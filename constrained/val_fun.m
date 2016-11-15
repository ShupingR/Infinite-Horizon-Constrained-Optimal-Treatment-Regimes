function val_pi = ...
         val_fun(state, center, sigma2, tau, which_reward, K, weight)
     
    all_phi = all_feature( state, center, sigma2, K );

    action = all_phi' * tau;
    
    val_pi = qfun( state, center, sigma2, action, which_reward, K, weight );
end