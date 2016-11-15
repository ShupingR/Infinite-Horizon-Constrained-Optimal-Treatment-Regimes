classdef SampleClass
   properties
      %state
      wellness
      tumorsize
      death
      %state range
      upp_wellness
      low_wellness
      upp_tumorsize
      low_tumorsize
      % action: dose
      action 
      % nxt_state
      next_wellness
      next_tumorsize
      next_death
      % reward
      negative_reward
      positive_reward
      % sample size
      sample_size
   end
end

