
M = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1; ...
    1 1 0 0; 0 1 1 0; 0 0 1 1; 0 1 0 1; 1 0 1 0; 1 0 0 1; ...
    1 1 1 0; 0 1 1 1; 1 0 1 1; 1 1 0 1; ...
    1 1 1 1];

for i = 1:15
    i
    index_vector = M(i, :);
    model = VAD_train_classifier(index_vector);
    models{index_vector(1) + 1, index_vector(2) + 1, index_vector(3) + 1, index_vector(4)+1} = model;        
end

save models