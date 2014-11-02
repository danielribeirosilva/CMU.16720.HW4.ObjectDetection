function [boundingBoxes] = batchDetectImageESVM(imageNames, models, params)

%running params
numCores = 4;
imageDir = '../../data/voc2007';

% Close the pools, if any
try
    fprintf('Closing any pools...\n');
    matlabpool close; 
catch ME
    disp(ME.message);
end

%open pool
fprintf('Will process %d files in parallel to compute detection ...\n',length(imageNames));
fprintf('Starting a pool of workers with %d cores\n', numCores);
matlabpool('local',numCores);


l = length(imageNames);

boundingBoxes = cell(l,1);
parfor i=1:l
    fprintf('Performing detection on %s\n', imageNames{i});
    I = imread(fullfile(imageDir, imageNames{i}));
    boundingBoxes{i} = esvm_detect(I,models,params);
end

%save('boundingBoxes.mat','boundingBoxes');

matlabpool close;


end