M = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1; ...
    1 1 0 0; 0 1 1 0; 0 0 1 1; 0 1 0 1; 1 0 1 0; 1 0 0 1; ...
    1 1 1 0; 0 1 1 1; 1 0 1 1; 1 1 0 1; ...
    1 1 1 1];

for i = 1:15
    i
    index_vector = M(i, :);
    VAD_noisy_data(index_vector);
end