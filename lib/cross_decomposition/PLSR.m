classdef PLSR < BaseEstimator & TransformerMixin & RegressorMixin
    % Partial least squares regression(PLSR).
    
    properties (GetAccess = 'public', SetAccess = 'public')
        % parameters
        n_components = 2; % Number of components to keep.
    end
    
    properties (GetAccess = 'public', SetAccess = 'private')
        % attributes
        x_loadings; % X block loadings vectors.
        y_loadings; % Y block loadings vectors.
        x_scores; % X, dimensionality-reduced
        y_scores; % Y, dimensionality-reduced
        x_rotations; % mapping from X to latents
        y_rotations; % mapping from Y to latents
    end
    
    methods
        % constructor
        function obj = PLSR(params)
            if nargin>0
                obj.set_params(params)
            end
        end
        
        % Fit the model with X and Y.
        function fit(obj,X,Y,~)
            [obj.x_loadings, obj.y_loadings, obj.x_scores, obj.y_scores] = ...
                plsregress(X, Y, obj.n_components);
        end
        
        % Apply the dimensionality reduction on X and Y (optional).
        function [X_new, Y_new] = transform(obj,X,Y)
            % check_is_fitted(obj) % as in sklearn
            
            if exists('Y', 'var')
            end
        end
        
        % Apply the dimension reduction learned from the training data
        function C = predict(obj,X)
%             C = X*obj.coef_ + obj.intercept_; TODO fix
        end
    end
end
