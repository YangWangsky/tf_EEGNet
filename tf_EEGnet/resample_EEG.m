% resample EEGs from 250Hz to 128Hz
clear;
clc;
load data.mat

EEGs_new = zeros(9, 288, 22, 256);
for i = 1:9
    for j = 1:288
        a = reshape(EEGs(1,1,:,:), 22, 500);
        EEGs_new(i,j,:,:) = resample(a', 128, 250)';
    end
end
EEGs = EEGs_new;
save('data_128Hz.mat', 'EEGs', 'labels');