M = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1; ...
    1 1 0 0; 0 1 1 0; 0 0 1 1; 0 1 0 1; 1 0 1 0; 1 0 0 1; ...
    1 1 1 0; 0 1 1 1; 1 0 1 1; 1 1 0 1; ...
    1 1 1 1];


for i = 1:15
    index_vector = M(i, :);
    
    VAD_training_data(index_vector);
end

