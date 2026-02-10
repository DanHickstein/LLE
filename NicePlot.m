function [ linehandle,axeshandle ] = NicePlot( TwoColumnInput,varargin )


M=TwoColumnInput;
if isempty(varargin)
linehandle=plot(M(:,1),M(:,2),'linewidth',2);
else
    linehandle=plot(M(:,1),M(:,2),varargin{1},'linewidth',2);
end
set(gca,'FontSize',22,'FontName','Calibri','linewidth',2);
axeshandle=gca;


end

