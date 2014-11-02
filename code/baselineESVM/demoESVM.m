addpath(genpath('../utils'));
addpath(genpath('../lib/esvm'));
I = imread('bus.jpg');
load('../../data/bus_esvm.mat');

params = esvm_get_default_params(); %get default detection parameters
detectionBoxes = esvm_detect(I,models,params);


figure; hold on; image(I); axis ij; hold on;
showboxes(I,  detectionBoxes);