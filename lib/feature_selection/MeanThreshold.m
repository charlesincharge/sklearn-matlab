classdef MeanThreshold < BaseEstimator & SelectorMixin
    % Feature selector that removes all low-mean features.
    % Useful for discarding low-rate Poisson features

    properties (GetAccess = 'public', SetAccess = 'public')
        % parameters
        threshold = 0; % (float) features with a training-set mean lower than this threshold will be removed.
    end
    
    properties (GetAccess = 'public', SetAccess = 'private')
        % attributes
        means; % (array, shape [n_features]) Means of individual features.
    end
    
    methods
        % constructor
        function obj = MeanThreshold(params)
            if nargin>0
                obj.set_params(params)
            end
        end
        
        % Fit the model with X.
        function fit(obj,X,~)
            dim = 1;
            obj.means = mean(X, dim);
            
            assert(~all(obj.means <= obj.threshold),...
                   'No feature in X meets the mean threshold %f', obj.threshold);
        end
        
    end
    
    methods (Access = 'protected')
        % Return the feature selection mask
        function mask = get_support_mask(obj)
            mask = obj.means > obj.threshold;
        end
    end
end
