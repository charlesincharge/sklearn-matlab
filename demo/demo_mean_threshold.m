% Remove features using a variance threshold

%%
close all
clear
rng('default');
rng(1); % for reproducibility

%% Load data
[X_train, X_test, y_train, y_test] = sklearn_data_poisson_processes();

%% Train, ignoring low-rate measurements of y
params = struct('threshold', 2);
mt = MeanThreshold(params);
ridge = Ridge;

pipeline = make_pipeline(mt, ridge);
pipeline.fit(X_train, y_train);

%% Predict
y_pred = pipeline.predict(X_test);
score = r2_score(y_test, y_pred);
fprintf('Regression score is %.2f\n',score);

%% Log diagnostics
fprintf('number of original features: %d\n', size(X_train, 2));
fprintf('number of features used: %d\n', nnz(mt.get_support()));

%% OUTPUT

figure;
scatter(y_pred, y_test);
xlabel('predicted');
ylabel('truth');
title('Ridge regression with MeanThreshold demo')
