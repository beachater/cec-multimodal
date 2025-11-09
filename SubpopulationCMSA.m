classdef SubpopulationCMSA < Subpopulation  % inherits the Subpopulation class
    % ----- Additional attributes of subpopulations in the core search is CMSA -----"""
    properties
        elite; % elite solutions
    end
    methods
        function subpop=SubpopulationCMSA(center,smean,stretch,popSize) % Constructor
            subpop@Subpopulation(center,smean,stretch,popSize); % initialize the properties of the parent class
            subpop.elite=Elite(numel(stretch)); % repetition of the subpopulation mean for the elite solutions of the subpopulation  
        end
        
        function sample_solutions(subpop,restart,archive,process,opt,problem)  
            % *** Sample new popualtion members and evaluate them ***  
            tempRedRatio=1; % For temoprary reduction of the taboo region if too many samples are rejected
            solNo=1; % solution number: counter for created taboo acceptable solutions

            % Preallocate values
            subpop.samples=Sampling(process.subpopSize,problem.dim); 


            % Sampling stage: generate taboo-acceptable solutions
            while solNo<=process.subpopSize
                %print(tempRedRatio)
                subpop.samples.s(solNo)=subpop.mutProfile.smean*exp(randn*process.coreSpecStrPar.tauSigma); % calculate individual step sizes
                subpop.samples.Z(solNo,:)=(subpop.mutProfile.rotMat*(subpop.mutProfile.stretch.*randn(1,problem.dim))')'; % vector Z

                
                subpop.samples.X(solNo,:)=subpop.center+subpop.samples.s(solNo)*subpop.samples.Z(solNo,:); % sampled solution
                subpop.repair_infeas(solNo,opt,problem); % relocate to the search range if outside

                %  now check if this sampled solution is taboo acceptable 
                acceptIt=subpop.is_taboo_acceptable(subpop.samples.X(solNo,:),tempRedRatio,opt); % check if the samples solution is taboo acceptable 
                if acceptIt % if it is taboo acceptable
                    solNo=solNo+1;
                else
                    tempRedRatio=tempRedRatio*opt.niching.redCoeff; % temperoray reductions of taboo region sizes if too many attempts failed consecutively
                end
            end  
        end
        
        
        
        function recombine(subpop,restart,archive,process,opt,problem) 
            % -------- Perform recombination when the core search is elitist CMSA -----------
            oldCenter=subpop.center;
            ind=subpop.samples.argsortWithElite;  % use sorted indexes from elite                    

            % update the center  
            subpop.center=process.recWeights*subpop.samples.X(ind(1:process.mu),:); % update the center of the subpopulation
            % Relocate Xmean to the closest point in the search range
            subpop.center=max([subpop.center;problem.lowBound]);
            subpop.center=min([subpop.center;problem.upBound]);

            % --------- update the best solujtion of the subpopulation ------------
            subpop.bestSol=subpop.samples.X(ind(1),:);
            subpop.bestVal=subpop.samples.f(ind(1)); 

            % ----------- update the global step size -------------
            correctionTerm=(geomean(subpop.samples.s)/subpop.mutProfile.smean)^opt.coreSearch.sigmaUpdateBiasImp; % Correction for rejection of some generated solutions (for unsymmetric distribution os individul step sizes)
            subpop.mutProfile.smean=prod(subpop.samples.s(ind(1:process.mu)).^process.recWeights) / correctionTerm;  % suggested  global step size by recombination

            % ---------- Update the covariance matrix ----------- 
            suggC=zeros(problem.dim,problem.dim); %suggested C based on the new solution evalautions % 
            for parNo=1:process.mu
                Z=(subpop.samples.X(ind(parNo),:)-oldCenter)/subpop.samples.s(ind(parNo)); % Z wrt the center in the just finished iteration
                suggC=suggC+process.recWeights(parNo)*(Z'*Z);
            end
            cc=1/process.coreSpecStrPar.tauCov; % learning rate for C 
            newC=(1-cc)*subpop.mutProfile.C+ cc*suggC; % update C
            subpop.mutProfile.C=0.5*(newC+newC'); % enforce symmetry
            [subpop.mutProfile.rotMat,tmp]=eig(subpop.mutProfile.C); % perform eigen decomposition to calculate rotMat and stretch
            subpop.mutProfile.stretch=sqrt(diag(tmp))';
            clear tmp
            subpop.mutProfile.Cinv= subpop.mutProfile.rotMat*diag(subpop.mutProfile.stretch.^(-2))*subpop.mutProfile.rotMat'; % inverce of the covariance matrix

            % ------------- Update elite solutions ----------------
            if process.coreSpecStrPar.numElt>0
                [~,surviveInd]=sort(subpop.samples.f);  
                limit1=sum(subpop.samples.isFeas)+numel(subpop.elite.val); % only feasible solutions  
                limit2=process.coreSpecStrPar.numElt; % another upper limit for the number of elite solutions
                actEltNum=floor(0.5+min(limit1,limit2)); % the number of elite solutions for the next iteration
                subpop.elite.sol=subpop.samples.X(surviveInd(1:actEltNum),:); 
                subpop.elite.val=subpop.samples.f(surviveInd(1:actEltNum)); 
                subpop.elite.Z=subpop.samples.Z(surviveInd(1:actEltNum),:); 
                subpop.elite.s=subpop.samples.s(surviveInd(1:actEltNum));
                subpop.elite.wasRepaired=subpop.samples.wasRepaired(surviveInd(1:actEltNum));
            end
        end
        function select(subpop,restart,archive,process,opt,problem)   
            % ----------- Perform selection -------------------  
            [~,subpop.samples.argsortNoElite]=sort(subpop.samples.f); % indexes of best non-elite samples

            % --- Append acceptable elite solutions to the seletion pool if the core search uses elitism and the elites are taboo acceptable ---- 
            if strcmp(opt.coreSearch.algorithm, 'CMSA') && (process.coreSpecStrPar.numElt>0) && numel(subpop.elite.val)>0 % if elitism is activated and there are elite solutions from the previous restart
                appendIt=ones(1,numel(subpop.elite.val)); % Preallocation
                % -------- only append elite solutions that are taboo acceptable: Find their indexes -----------
                for eltNo=1:numel(subpop.elite.val)
                    appendIt(eltNo)=subpop.is_taboo_acceptable(subpop.elite.sol(eltNo,:),1,opt); % only taboo elite solutions can be added (no temporary shrinkage of taboo regions is allowed in this case)
                end
                appendIt=find(appendIt);
                % ------ append the approved elite solutions to the selection pool -------
                subpop.samples.X=[subpop.samples.X;subpop.elite.sol(appendIt,:)];
                subpop.samples.Z=[subpop.samples.Z;subpop.elite.Z(appendIt,:)];
                subpop.samples.s=[subpop.samples.s,subpop.elite.s(appendIt)];
                subpop.samples.f=[subpop.samples.f,subpop.elite.val(appendIt)];
                subpop.samples.wasRepaired=[subpop.samples.wasRepaired,subpop.elite.wasRepaired(appendIt)];
            end  % if

            % ----- For performing elite selection 
            [~,subpop.samples.argsortWithElite]=sort(subpop.samples.f); % indices of best solutions (sorted)
        end % function

    end
end
        