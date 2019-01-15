A = magic(4);

index_vector = [1 0 0 1];
leave_out_index = 0;
for i =1:length(index_vector)
    
    if index_vector(i) == 0
        A(:, i-leave_out_index) = [];
        leave_out_index = leave_out_index + 1;
        
    end
end

A