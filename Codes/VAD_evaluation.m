% noise number (0, 1) determines do we evaluate test set or noisy set 

function [MER, er1, er2] = VAD_evaluation(index_vector, models, noise_switch)

    % Load files
    if noise_switch == 0
        i = num2str(index_vector);
        name_part = strcat('test_data.', i);
        final_features = strcat(name_part, '.csv');
    
        test_features = csvread(final_features);   
        test_labels = csvread('test_labels.csv');
    else
         i = num2str(index_vector);
        name_part = strcat('test_noisy_data.', i);
        final_features = strcat(name_part, '.csv');
    
        test_features = csvread(final_features);   
        test_labels = csvread('test_labels.csv');
    end
             
    predicted_labels = predict(models{index_vector(1) + 1, index_vector(2) +1, index_vector(3) + 1, index_vector(4)+1}, test_features);
    [MER, er1, er2] = evaluation(predicted_labels, test_labels)

    
end

