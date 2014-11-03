function Iout = seg2im( segs )
% CV F14 - Provided Code
%
% Visualizing segments as randomly colored image
%
% Input:
%   segs - segmentation output, should be of size H x W where H is the
%          height, and W is the width
% Output:
%   Iout - the randomly colored segmenation
%
% Created by Wen-Sheng Chu (Sep-20-2014)

Iout   = zeros(numel(segs),3);
segall = segs(:);
for i = unique(segall)'
	ind = segall==i;
	Iout(ind,:) = repmat(randi(255,1,3),sum(ind),1);
end
Iout = uint8(reshape(Iout,[size(segs),3]));
