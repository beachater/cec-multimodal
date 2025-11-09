classdef GlobalMinimaData < handle
    % Information about the global minima. This inoformation is not used during optimization.
    % It is only used for performance evaulation."""
    properties
        Rnich=nan; % the niche radius
        Ngmin=nan; % The number of global minima
        val=nan; % The global minimum value
        X=nan; % The global minima
    end
end