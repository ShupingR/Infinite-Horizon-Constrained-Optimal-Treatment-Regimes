algorithm = 1;
maxiterations = 100;
epsilon = 0.001;
samples = train_sample_collect(1000, 7, 111);
initial_policy_weights = ones(5*5 , 1);
lspi(algorithm, maxiterations, epsilon, samples, initial_policy_weights)