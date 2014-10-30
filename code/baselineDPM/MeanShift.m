%data: N x (F + 1)
function [CCenters, CMemberships] = MeanShift(data, bandwidth, stopThresh)

w = data(:,end);
X = data(:,1:end-1);
[N,F] = size(X);


%mean shift for each points
Xfinal = zeros(N,F);
for i=1:N
    
    %fprintf('I: %i\n', i);
    
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
        if sum(idx)==1 %avoid bug of bsxfun for 1 point
            Xmean = X(idx,:);
        elseif sum(idx)>1
            Xmean = sum(bsxfun(@times,X(idx,:),w(idx,:))) / sum(w(idx,:)) ;
        else
            break;
        end
        
        if i==617
            fprintf('----------\n');
            disp(sum(idx));
            disp(X(idx,:));
            disp(Xi);
            disp(Xmean);
        end
        
        %update
        XiOld = Xi;
        Xi = Xmean;
        
        %compute diff
        diff = norm(Xi - XiOld);
        
        %fprintf('considered points: %i\n', sum(idx));
        %fprintf('diff: %d\n\n', diff);
        
    end
    
    Xfinal(i,:) = Xi;
    
end


%get the cluster centers

distThresh = norm(max(X)-min(X))/100;
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


end