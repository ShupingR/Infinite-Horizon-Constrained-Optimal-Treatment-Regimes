function qval = ...
         qfun(state, center, sigma2, action, which_reward, K , weight)
     if( any(action == [0, 0.2, 0.4, 0.6, 0.8, 1]) )
         qval = feature( state, center, sigma2, ...
                         action, which_reward, K )' * weight;
     else
        display( 'please check action input' );
     end           
end