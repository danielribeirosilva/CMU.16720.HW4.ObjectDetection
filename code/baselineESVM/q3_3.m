%% setting path and load model
addpath(genpath('../utils'));
addpath(genpath('../lib/esvm'));
load('../../data/bus_data.mat');
load('../../data/bus_esvm.mat');


%get default params
params = esvm_get_default_params();


lpo_values = 2:10;
rec_results = cell(1,length(lpo_values));
prec_results = cell(1,length(lpo_values));
ap_results = zeros(1,length(lpo_values));

i=1;
for lpo=lpo_values
    
    fprintf('Computing Boxes for LPO=%i...\n',lpo);
    
    %adjust param
    params.detect_levels_per_octave = lpo;

    %get boxes
    [boundingBoxes] = batchDetectImageESVM(gtImages, models, params);
    
    %save boxes
    file_name = ['EsvmBoxes_', num2str(lpo),'.mat'];
    save(file_name,'boundingBoxes');

    %get performance
    [rec,prec,ap] = evalAP(gtBoxes,boundingBoxes);
    
    %store results
    rec_results{i} = rec;
    prec_results{i} = prec;
    ap_results(i) = ap;
    
    %counter 
    i=i+1;
end


%plot results

%AP
figure;
plot(lpo_values,ap_results);
title('AP vs. LPO');
xlabel('LPO');
ylabel('AP');
