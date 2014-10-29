%data: N x (F + 1)
function [CCenters, CMemberships] = MeanShift(data, bandwidth, stopThresh)

w = data(:,end);
X = data(:,1:end-1);
[N,F] = size(X);


%mean shift for each points
Xfinal = zeros(N,F);
for i=1:N
    
    %do mean shift for point Xi
    Xi = X(i,:);
    %initialize diff
    diff = stopThresh + 1;
    
    while diff > stopThresh
        
        dist = bsxfun(@minus,X,Xi);
        dist = sqrt(sum(dist.^2,2));
        
        %get poiints that will participate in update
        idx = dist < bandwidth/2;
        
        %compute mean
        if sum(idx)>0
            Xmean = sum(bsxfun(@times,X(idx,:),w(idx,:))) / sum(w(idx,:)) ;
        else
            Xmean = zeros(1,F);
        end
        
        %update
        XiOld = Xi;
        Xi = Xi + Xmean;
        
        %compute diff
        diff = norm(Xi - XiOld);
        
        disp(Xi);
        disp(Xmean);
        fprintf('considered points: %i\n', sum(idx));
        fprintf('diff: %d\n\n', diff);
        
    end
    
    Xfinal(i,:) = Xi;
    
    break;
    
end


%get the cluster centers

distThresh = 0.05;
CMemberships = zeros(N,1);
CCenters = Xfinal(1,:);
CMemberships(1) = 1;
for i=2:N
    
    Xi = Xfinal(i,:);
    %distance to existing centers
    [closestCenterDist, pos] = min(pdist2(CCenters,Xi));
    if closestCenterDist <= distThresh
        CMemberships(i) = pos;
    else
        CCenters = [CCenters; Xi];
        CMemberships(i) = size(CCenters,1);
    end
    
end




CCenters = [];
CMemberships = [];


end