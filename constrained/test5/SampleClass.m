classdef SampleClass
   properties
      %state
      wellness
      tumorsize
      death
      phi 
      % 6 by 6*K by 2 array cube , K is dimension for the feature
          % 0 for postion not for the action
%       %state range
%       upp_wellness
%       low_wellness
%       upp_tumorsize
%       low_tumorsize
      % action: dose
      action 
      % nxt_state
      next_wellness
      next_tumorsize
      next_death
      next_phi 
      % 6 by 6*K by 2 array cube , K is dimension for the feature
      % 0 for postion not for the action
      % reward
      negative_reward
      positive_reward
      % sample size
      sample_size
   end
end

