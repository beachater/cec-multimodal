% main file to run the AMLP-RS-CMSA-ESI
% contact Ali Ahrari (aliahrari1983@gmail.com) for queries
% last check on 20 July 2022

clear all
 
%% Parameter setting 
pid=1; % problem ID (1-24)
seedNo=1; % run number (random seed No)

%% No need to change anything below
if ~exist('result','dir')
    mkdir('result')
end
addpath('dmmops');addpath('dmmops/problem');addpath('dmmops/problem/Data');addpath('dmmops/problem/Sub_f') 

%% Creating the problem object
problem=OptimProblemDMMOP(pid); %create the problem object
problem.knownChanges=false; % unknown changes
problem.set_problem_data();

%% Options for optimization
coreSeachMethod='CMSA';
opt=OptimOption(problem,coreSeachMethod); % create the optimization object
opt.stopCr.merge.windowSizeCoeff=1;   
opt.coreSearch.iniSubpopSizeCoeff=9;
opt.coreSearch.finSubpopSizeCoeff=9;
opt.archiving.tolFunArch=.001;
opt.stopCr.tolHistFun=0.00001;
opt.coreSearch.iniSigCoeff=1;
opt.coreSearch.muToPopSizeRatio=0.5;
opt.coreSearch.maxIniSigma=0.1;
opt.coreSearch.eltRatio=0;
opt.dyna.maxPL=10;
opt.dyna.recStepSizeCoeff=0.3;
opt.dyna.chCheckFr=30;
outfile=['prb' num2str(pid) '_run' num2str(seedNo) '.mat'];
driver(opt,problem,seedNo,outfile); % run the optimization process, and store detailed information in the dictionary data
 
 


