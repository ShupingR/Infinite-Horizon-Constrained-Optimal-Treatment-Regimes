algorithm = 1;
maxiterations = 100;
epsilon = 0.001;
[ sample_pos, sample_neg ] = train_sample_collect(1000, 7, 111);
initial_policy_weights = ones(5*5 , 1);
lspi(algorithm, maxiterations, epsilon, sample_pos, initial_policy_weights)
lspi(algorithm, maxiterations, epsilon, sample_neg, initial_policy_weights)