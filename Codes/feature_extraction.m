function feature_matrix = feature_extraction(feature_matrix, frame_matrix, Fs)


    
    END = size(feature_matrix, 1);
    for iFrame = 1:size(frame_matrix,2)
        
                                                               
        % Implement zero-crossing rate (ZCR) computation
        feature_matrix(END + iFrame, 1) = ex3_zcr_solution(frame_matrix(:,iFrame));
       
        
        % Implement energy computation    
        feature_matrix(END + iFrame, 2)  = ex3_energy_solution(frame_matrix(:,iFrame));
        
      
        
        if feature_matrix(END + iFrame, 2) == 0
            feature_matrix(END + iFrame, 3) = 0;
        else
            
            % Implement one-lag autocorrelation computation
            feature_matrix(END + iFrame, 3) = ex3_one_lag_autocorrelation_solution(frame_matrix(:,iFrame));
        end
             
        % f0 searching range
        f0_min = 80;
        f0_max = 300;
        
        if feature_matrix(END + iFrame, 2) == 0
            feature_matrix(END + iFrame, 4) = 0;
        else            
         %   Get the peak values
            [~, feature_matrix(END + iFrame, 4)] = ex2_fundf_cepstrum_solution(frame_matrix(:,iFrame),Fs,f0_min,f0_max,0);
        end
        
        
        
    
    end
    
end
 
    