addpath(genpath('../utils'));
addpath(genpath('../lib/esvm'));
I = imread('peppers.png');
load('../../data/bus_esvm.mat');

params = esvm_get_default_params(); %get default detection parameters
detectionBoxes = esvm_detect(I,models,params);
