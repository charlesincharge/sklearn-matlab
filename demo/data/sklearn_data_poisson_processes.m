function [X_train, X_test, y_train, y_test] = sklearn_data_poisson_processes()
% Generate a data set containing a Poisson process in each feature

%% PARAMETERS
test_size = 1/3;
num_times = 300;
num_processes = 50;
lambda_scale = 10;

%% GENERATE
% time-varying underlying rate is the ground-truth label
lambdas = lambda_scale * rand(num_times, 1);

% Each feature/process has a rate scaled from the underlying rate
rate_scales = rand(1, num_processes);

% Scale time-varying lambdas for each process
rates = lambdas * rate_scales;
events = poissrnd(rates);

% Split datasets by time-points
[X_train, X_test, y_train, y_test] = train_test_split(events,lambdas,test_size);