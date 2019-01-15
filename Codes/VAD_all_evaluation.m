M = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1; ...
    1 1 0 0; 0 1 1 0; 0 0 1 1; 0 1 0 1; 1 0 1 0; 1 0 0 1; ...
    1 1 1 0; 0 1 1 1; 1 0 1 1; 1 1 0 1; ...
    1 1 1 1];
close all;
load models;

noise_switch = 1;

for i = 1:15
    index_vector = M(i, :);
    if i <= 4
        
        [MER, er1, er2] = VAD_evaluation(index_vector, models, noise_switch);
        one_par_data(:, i) = [MER, er1, er2]'*100;        
    
    
    elseif i <= 10
         
        [MER, er1, er2] = VAD_evaluation(index_vector, models, noise_switch);
        two_par_data(:, i-4) = [MER, er1, er2]'*100;  
        
    elseif i <= 14
        
        [MER, er1, er2] = VAD_evaluation(index_vector, models, noise_switch);
        three_par_data(:, i-10) = [MER, er1, er2]'*100;  
        
    else
        
       [MER, er1, er2] = VAD_evaluation(index_vector, models, noise_switch)
       four_par_data(:, i-14) = [MER, er1, er2]'*100;  
    end
end

%% Visualize results

x = [1 2 3 4];
plot(x, one_par_data(1, :), 'o', x, one_par_data(2, :), 'o', x,  one_par_data(3, :), 'o');
title('Error rates of classification with 1-parameter models');
xlabel('Parameter set');
ylabel('Error rate (%)');
grid on;
legend('MER', 'NDAS-error', 'SDAN-error');

figure;

x = [1 2 3 4 5 6];
plot(x, two_par_data(1, :), 'o', x, two_par_data(2, :), 'o', x,  two_par_data(3, :), 'o');
title('Error rates of classification with 2-parameter models');
xlabel('Parameter set');
ylabel('Error rate (%)');
grid on;
legend('MER', 'NDAS-error', 'SDAN-error');

figure; 

x = [1 2 3 4];
plot(x, three_par_data(1, :), 'o', x, three_par_data(2, :), 'o', x,  three_par_data(3, :), 'o');
title('Error rates of classification with 3-parameter models');
xlabel('Parameter set');
ylabel('Error rate (%)');
grid on;
legend('MER', 'NDAS-error', 'SDAN-error');







