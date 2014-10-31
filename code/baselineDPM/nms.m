function [refinedBBoxes ] = nms(bboxes, bandwidth, K)

%threshold
stopThresh = bandwidth*0.001;

%initialize 
refinedBBoxes = [];

%normlize weights
w = bboxes(:,end);
w = 1 + (w - min(w(:)))/ (max(w(:)) - min(w(:)));
bboxes(:,end) = w;

%perform Mean Shift
[CCenters, CMemberships] = MeanShift(bboxes, bandwidth, stopThresh);

%Non-Max Suppression
for i = 1:size(CCenters,1)
    
    %get cancidate boxes for given cluster
    candidates = bboxes(CMemberships==i,:);
    
    %get box with highest score
    [~, pos] = max(candidates(:,end));
    
    %add to result
    refinedBBoxes = [refinedBBoxes; candidates(pos,:)];
    
end

%limit to K 
if K<size(refinedBBoxes,1)
    [~,I]=sort(refinedBBoxes(:,end));
    refinedBBoxes=refinedBBoxes(I,:); 
    refinedBBoxes = refinedBBoxes(end-K+1:end,:);
end
