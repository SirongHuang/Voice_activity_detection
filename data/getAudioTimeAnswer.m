function [ y ] = getAudioTimeAnswer( keys, utterance_index, length, fs)
%GETAUDIOTIMEANSWER keys straight from importfile.m, utterance_index the
%index into keys, length in samples, fs=sampling rate

x = zeros(length,1);
flipswitch = 1;
for time = cell2mat(keys(utterance_index,2:end))
    if isnan(time)
        break
    end
    x(round(time*fs):end) = flipswitch;
    if flipswitch == 1;
        flipswitch = 0;
    else
        flipswitch = 1;
    end
end
y=x;