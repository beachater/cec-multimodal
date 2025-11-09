classdef Prediction
    methods (Static)
        function [xhatNow,estPreErrNowNorm,bestLevel]=AMLP(xopt,maxLevel)
            % Inputs:
                % xopt:
                    % The time history of a particular global optimum (or its 
                    % approximation) at different time steps (time steps 0, 1, ..., t). 
                    % It is a matrix of size N*D, in which D is the problem dimension 
                    % and N is the number of solutions. Only the last maxLevel rows are
                    % required, so N can be less than t. However, this code 
                    % automatically discard older solution if they are provided. 
                % maxLevel:
                    % the maximum level considered for prediction (e.g. 20)
            % Outputs:
                % xhat:
                    % the predicted solution (global optimum) by each prediction level 
                    % in time step t+1 (currentTimeStep)
                % estPreErr:
                    % Estimated prediction error for time step t+1 from each prediction
                    % level
                % bestLevel:
                    % The most reliable prediction level for time step t+1, which can
                    % be used to define the center and variation of the seed 
                    % population for time step t+1

            [T,D]=size(xopt); 
            %% Calculate the prediction and the prediction error from all requested and possible levels
            [xhat,preErr,maxLevelPrime]=Prediction.MLP(xopt,maxLevel); % see MLP.m for explanation of the inputs/outputs
            estPreErrNowNorm=nan(1,maxLevelPrime); % preallocation of the norm of the estimated prediction error for the prediction in the new time step
            xhatNow=nan(maxLevelPrime,D); % preallocation of the predicted optimum for the new time step 
            %% store the prediction and the estimated error from each level
            for L=1:maxLevelPrime
                estPreErrNowNorm(L)=norm(preErr{L}(end,:)); 
                xhatNow(L,:)=xhat{L}(end,:);
            end
            %% return the outcome of the best prediction level
            if T==1 % no other prediction can be made for the second time step
                bestLevel=1;
                estPreErrNowNorm=nan; % No estimation for the prediction error can be calculated
            else
                [~,bestLevel]=min(estPreErrNowNorm);
                % now if the following two conditions are satisifed, consider the next level as the best level 
                if (bestLevel==maxLevelPrime-1) && (maxLevelPrime==T)
                    bestLevel=bestLevel+1;
                end
            end
        end
 


        function [xhat,preErr,maxLevelPrime]=MLP(xopt,maxLevel)
            % Inputs:
                % xopt:
                    % The time history of a particular global optimum at time steps 1, ..., T 
                    % It is a matrix of size T*D, in whiche maximum prediction level is
                    % automatically set to T.  
            % Outputs:
                % xhat:
                    % the predicted solution (global optimum) by each prediction level 
                    % for each level and each time step   
                % preErr:
                    % prediction error for each level and each time step

            [t,D]=size(xopt); %  the number of solution and the problem dimension.  
            t=t-1; % the just finished time step 

            maxLevelPrime=min(maxLevel,t+1); % you cannot/should not calculate prediction levels higher than this
            t_recent=max(0,t-maxLevel); % solutions before this time steps are not used for prediction

            xhat=cell(1,maxLevelPrime); % predicted global optimum (preallocation) 
            preErr=cell(1,maxLevelPrime);  % prediction error (preallocation)

            %% calculate the prediction solution and the corresponding prediction error for each time step at level 1
            L=1; % prediction level
            xhat{L}=nan(t+2,D); % preallocation for prediction solution
            preErr{L}=nan(t+1,D);
            for tau=t_recent:t
                xhat{L}(tau+2,:)=xopt(tau+1,:); % Predicted solution for time step tau
                if tau<t
                    preErr{L}(tau+2,:)=xopt(tau+2,:)-xhat{L}(tau+2,:);
                end
            end

            %% calculate predicted solution and predicted error for higher levels  
            for L=2:maxLevelPrime
                xhat{L}=nan(t+2,D); % preallocation for prediction solution
                preErr{L}=nan(t+1,D); % preallocation for prediction solution

                for tau=t_recent:t
                    if tau>=L-1
                        xhat{L}(tau+2,:)=xhat{L-1}(tau+2,:)+preErr{L-1}(tau+1,:);
                        if tau<t
                            preErr{L}(tau+2,:)=xopt(tau+2,:)-xhat{L}(tau+2,:);
                        end
                    end
                end
            end 
            % The following lines deals with an exceptional case, in which the 
            % predicted solution from the Level maxLevelPrime is available but no 
            % estimate for this prediction is available. In this case, the value of the
            % lower level is used
            if (maxLevelPrime==t+1) && (maxLevelPrime>1) 
                preErr{maxLevelPrime}(t+1,:)=preErr{maxLevelPrime-1}(t+1,:);
            end
        end
    end
end

