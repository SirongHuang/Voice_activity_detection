
M = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 0 1];

for i = 1:4
    M(i, :) = index_vector;
    model = VAD_train_classifier(index_vector);
    models{index_vector(1) + 1, index_vector(2) + 1, index_vector(3) + 1, index_vector(4)}        
end

save models