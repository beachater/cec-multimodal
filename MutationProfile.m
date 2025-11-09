classdef MutationProfile < handle
    % ------ Mutation profile of the subpopulation ------- 
    properties
        smean % step size
        stretch % scale factor along main axis of C
        C % Covariane matrix
        Cinv % inverse of covariance matrix 
        rotMat % rotation matrix (derived by eigen decomposition of C)
    end
    methods
        function mutProfile=MutationProfile(smean,stretch)
            mutProfile.smean=smean; 
            mutProfile.stretch=stretch;  
            mutProfile.C=diag(mutProfile.stretch.^2);  
            mutProfile.Cinv=diag(mutProfile.stretch.^(-2));   
            mutProfile.rotMat=eye(numel(stretch));  
        end
    end
end