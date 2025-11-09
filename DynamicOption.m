classdef DynamicOption
    properties
    % """ Options peculiar to dynamic problems """
        benchSolSizeCoeff=1 % multiplied by population size to define the number of becnhmark solutions stored for change detection mechanism
        recStepSizeCoeff=.5
        tolChangeF=1e-5 % change detection tol 
        predictMethod="AMLP" % "AMLP" or "FLP"
        chCheckFr=10
        maxPL=4
    end
end