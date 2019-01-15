
function model = VAD_train_classifier(index_vector)

i = num2str(index_vector);
name_part = strcat('training_data.', i);
features = strcat(name_part, '.csv');

training_labels = csvread('training_labels');
training_features = csvread(features);

% Support vector machine
model = fitcsvm(training_features, training_labels);

end







