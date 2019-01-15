function [MER, er1, er2] = evaluation(predicted_labels, labels)

MER = 0;
er1 = 0;
er2 = 0;

for i = 1:length(labels)
    
    if predicted_labels(i) ~= labels(i)
            MER = MER + 1;
            if predicted_labels(i) == 1
                er1 = er1 + 1;
            else
                er2 = er2 + 1;
            end
    end
end

MER = MER/length(labels);   
er1 = er1/length(labels);
er2 = er2/length(labels);
