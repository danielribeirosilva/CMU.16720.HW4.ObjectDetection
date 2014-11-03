function [f,d] = dsift(I)
% CV F14 - Provided Code
%
% Create an image overlayed with the boundary map
% Credit to VLfeat (http://www.vlfeat.org/ by A. Vedaldi and B. Fulkerson)
%
% Input:
%   I     - input image
%
% Output:
%   f     - location of central pixel for each SIFT extracted
%   d     - descriptor of each SIFT descriptor
%
% Created by Wen-Sheng Chu (Sep-20-2014)

if ismac
	if strcmpi(computer,'maci64')
		addpath('external/mex/mexmaci64');
	else
		addpath('external/mex/mexmaci32');
	end
elseif isunix
	if strcmpi(computer,'glnxa64')
		addpath('external/mex/mexa64');
	else
		addpath('external/mex/mexglx');
	end
elseif ispc
	if strcmpi(computer,'pcwin64')
		addpath('external/mex/mexw64');
	else
		addpath('external/mex/mexw32');
	end
end

% conver to lab
Ilab = rgb2lab(I);

% smooth image
binSize = 8 ;
magnif = 3 ;
Is1 = vl_imsmooth(single(Ilab(:,:,1)), sqrt((binSize/magnif)^2 - .25)) ;
Is2 = vl_imsmooth(single(Ilab(:,:,2)), sqrt((binSize/magnif)^2 - .25)) ;
Is3 = vl_imsmooth(single(Ilab(:,:,3)), sqrt((binSize/magnif)^2 - .25)) ;

% extract sift on each channel
[f,d1] = vl_dsift(single(Is1),'size',1,'fast');
[f,d2] = vl_dsift(single(Is2),'size',1,'fast');
[f,d3] = vl_dsift(single(Is2),'size',1,'fast');
f = round(f);
d = [d1;d2;d3];