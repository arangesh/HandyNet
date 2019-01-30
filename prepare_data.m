clc;
clear;
close all;

%%
root = '/path/to/dataset';
train_path = fullfile(root, 'train');
val_path = fullfile(root, 'val');
train_objects = importdata(fullfile(train_path, 'objects.txt'));
val_objects = importdata(fullfile(val_path, 'objects.txt'));
train_seq_list = dir(fullfile(train_path, 'seq*'));
train_seq_list = {train_seq_list.name};
val_seq_list = dir(fullfile(val_path, 'seq*'));
val_seq_list = {val_seq_list.name};

%% Training Set
train_text = fopen(fullfile(root, 'train.txt'), 'w');
idx = cell(length(train_seq_list), 1);
train_output = {};

for i = 1:length(train_seq_list)
    filenames = dir(fullfile(train_path, train_seq_list{i}, 'smooth_depth', '*.mat'));
    filenames = {filenames(:).name}';
    
    idx{i} = randperm(length(filenames));
    for j = 1:length(idx{i})
        train_output{end+1} = sprintf('%s,%s,%s\n', train_seq_list{i}, filenames{idx{i}(j)}(1:end-4), train_objects{i});
    end
end

for i = randperm(length(train_output))
    fprintf(train_text, train_output{i});
end
fclose(train_text);
fprintf('Done with training split.\n\n');

%% Validation Set
val_text = fopen(fullfile(root, 'val.txt'), 'w');
idx = cell(length(val_seq_list), 1);
val_output = {};

for i = 1:length(val_seq_list)
    filenames = dir(fullfile(val_path, val_seq_list{i}, 'smooth_depth', '*.mat'));
    filenames = {filenames(:).name}';
    
    idx{i} = randperm(length(filenames));
    for j = 1:length(idx{i})
        val_output{end+1} = sprintf('%s,%s,%s\n', val_seq_list{i}, filenames{idx{i}(j)}(1:end-4), val_objects{i});
    end
end

for i = randperm(length(val_output))
    fprintf(val_text, val_output{i});
end
fclose(val_text);
fprintf('Done with validation split.\n\n');

%%
H = 424;
W = 512;
tot_train_files = length(train_output);
denom = tot_train_files*H*W;
raw_depth_mean = 0;
smooth_depth_mean = 0;

for i = 1:length(train_seq_list)
    filenames = dir(fullfile(train_path, train_seq_list{i}, 'raw_depth', '*.mat'));
    filenames = {filenames(:).name}';
    
    for j = 1:length(filenames)
        load(fullfile(train_path, train_seq_list{i}, 'raw_depth', filenames{j}));
        raw_depth_mean = raw_depth_mean + sum(log(currdata(currdata ~= 0)))/(denom*length(currdata(currdata ~= 0))/(H*W));
        load(fullfile(train_path, train_seq_list{i}, 'smooth_depth', filenames{j}));
        smooth_depth_mean = smooth_depth_mean + sum(log(currdata(:)))/denom;
    end
    fprintf('Done with %s!\n', train_seq_list{i});
end
fprintf('The mean for the (log) raw depth dataset is: %f\n', raw_depth_mean);
fprintf('The mean for the (log) smooth depth dataset is: %f\n', smooth_depth_mean);
