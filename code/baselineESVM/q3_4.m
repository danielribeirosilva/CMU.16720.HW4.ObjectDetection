%% setting path and load model
addpath(genpath('../utils'));
addpath(genpath('external'));
addpath(genpath('../lib/esvm'));
load('../../data/bus_data.mat');
load('../../data/bus_esvm.mat');

%params
imgDir = '../../data/voc2007/';
patchSize = 100;
alpha = 100;
K = 100;

%data
filterBank = createFilterBank();
F = 3*size(filterBank,1);
ISquareBoxes = cell(1,length(models));
filterResponses = [];
originalImage = [];

%get images from bounding boxes
for i=1:length(models)
   
    fprintf('i=%i/%i\n',i,length(models));
    
    %read image
    I = imread([imgDir,models{i}.I]);
    
    %get bounding box image
    imgBox = models{i}.gt_box;
    IBox = I(imgBox(2):imgBox(4),imgBox(1):imgBox(3),:);
    
    %resize to standard size
    IBoxSquare = imresize(IBox, [patchSize patchSize]);
    
    %store image
    ISquareBoxes{i} = IBoxSquare;
    fprintf('numel for %i: %i\n',i,numel(ISquareBoxes{i}));
    
    %compute filter responses
    imgResponses = extractFilterResponses(IBox,filterBank);
    
    %select alpha responses
    selectedRows = randperm(size(imgResponses,1),min(alpha,size(imgResponses,1)));
    selectedResponses = imgResponses(selectedRows,:);
    
    %store sampled responses
    filterResponses = [filterResponses; selectedResponses];
    originalImage = [originalImage; i*ones(numel(selectedRows),1)];
    
end

save('filterResponses.mat', 'filterResponses', 'originalImage', 'ISquareBoxes');

%cluster
[clusterBelonging, ~, ~, distanceToCenters] = kmeans(filterResponses, K, 'EmptyAction', 'drop');

%get cluster average image and best representant
clusterRepresentants = zeros(1,K);
avgImages = cell(1,K);
for i=1:K
    
    %get best representant
    [~,pos] = min(distanceToCenters(:,i));
    clusterRepresentants(i) = originalImage(pos);
    
    %get average image
    idx_imgs = originalImage(clusterBelonging==i);
    Iavg = zeros(patchSize,patchSize,3);
    for j=1:numel(idx_imgs)
        Iavg = Iavg + double(ISquareBoxes{idx_imgs(j)});
    end
    Iavg = uint8(Iavg/numel(idx_imgs));
    avgImages{i} = Iavg;
    
end

%evaluate reduced model
reduced_models = models(unique(clusterRepresentants));
params = esvm_get_default_params();
params.detect_levels_per_octave = 3;
[boundingBoxes] = batchDetectImageESVM(gtImages, reduced_models, params);
[~,~,ap] = evalAP(gtBoxes,boundingBoxes);

fprintf('Average Precision for K=%i: %d\n',K,ap);

imdisp(avgImages);

