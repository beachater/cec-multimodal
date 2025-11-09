% calculate PR for a single run for different function tolerances
% Last update by Ali Ahrari on 20 July 2022 
clear all
tic

%% Specify the result file and performance indicator parameters
filename='prb1_run1.mat';

tolf=[ .001 .0001 .00001 ]; % target precision 
Rnich=0.05; % distance criterion for detection of a global optimum

%% Do not change anything below
restoredefaultpath;addpath(pwd)
addpath('dmmops');addpath('dmmops/problem');addpath('dmmops/problem/Data');addpath('dmmops/problem/Sub_f') 
addpath('result')

%% load the saved data
load(filename,'process','problem')
Nenv=problem.extProb.maxEnv;
chFr=problem.extProb.freq;
pr=-1*ones(numel(tolf),Nenv);
for t=1:Nenv
    xopt=problem.extProb.his_o{t,3};
    fopt=-max(problem.extProb.his_of{t,3});
    rangeFE=problem.extProb.his_o{t,2}+[0,problem.extProb.freq]; % range of relevant function evaluations
    archive=process.dynamics.endArchive{t}; % reported solutions for this time step
    pr0=zeros(1,numel(tolf));
    for tolFind=1:numel(tolf)
       [wasFound,pr0(tolFind)]=calc_pr_dmmop(archive,xopt,fopt,rangeFE,tolf(tolFind),Rnich);
    end
    pr(:,t)=(pr0);
end
mpr=mean(pr,2);
disp('MPR for the selected accuracy:')
disp(mpr')

%% Calculate PR for an arbitrary time step
function [wasFound,pr]=calc_pr_dmmop(archive,xopt,fopt,rangeFE,tolf,Rnich)
    % find the indexes of relevant solutions (found in the relevant time and
    % have a fitness sufficiently close to the global optimum value)
    ind=archive.foundEval2>=rangeFE(1) & (archive.foundEval2<=rangeFE(2)) & ((archive.value-fopt)<tolf );
    if sum(ind)==0 % no good solution
        wasFound=false;
        pr=0;
    else
        repX=archive.solution(ind,:); % reported solutions
        clear ind
        wasFound=zeros(1,size(xopt,1)); % check if each global optimum has been found 
        for k=1:size(repX,1)
            % check the distance metric
            dis=pdist2(repX(k,:),xopt);
            [minDis,ind]=min(dis);
            if minDis<Rnich 
                wasFound(ind)=wasFound(ind)+1;
            end
        end
        pr=mean(wasFound>0);
    end
end
        
