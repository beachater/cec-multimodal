
classdef UtilityMethods
    methods (Static)
        function p = lin_prctile(data,q)
            n=numel(data);
            if n==1
                p=data(1);
            else
                data=sort(data);
                x=(0:(numel(data)-1))/(numel(data)-1)*100; % x-axis
                p=interp1(x,data,q);
            end
        end
        function p2p=peak2peak(x)
            p2p=max(x)-min(x);
        end
    end
end


