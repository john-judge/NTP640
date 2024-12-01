function Chap17_PETH(neuronNum)
%Chapter 17 - Matlab for Neuroscientists 2e
%PETH
%Nov 15, 2013
load('NeuralData')


%choose a default neuron if none specified
if ~exist('neuronNum')
    neuronNum=129;
end

graph = [6 3 2 1 4 7 8 9]; %sets the order to plot graphs in
figure
hold on

%set edges of the PET histogram
edgesPeri = -1:.02:1;  % 20 ms bins
psth=zeros(length(edgesPeri),8);

%compute PETH for each direction
for i=1:8
    indDir=find(direction==i); %find trials in a given direction
    numTrials(i)=length(indDir);
    
    for j =1:numTrials(i)
        %pick one of the following:
        centerTime=go(indDir(j)); %to center on start of movement time
        %centerTime=instruction(indDir(j)); %to center on instruction time

        allTimes=unit(neuronNum).times-centerTime; %center spike times
        spikeTimes=allTimes(allTimes>-1&allTimes<1); %pick 2 seconds around center time

        %add histogram of present trial to PETH
        if ~isempty(spikeTimes)
            psth(:,i)=psth(:,i)+histc(spikeTimes,edgesPeri,1);
        end
    end  
    %divide by the number of trials & bin size for a mean firing rate
    psthNorm(:,i)=psth(:,i)/numTrials(i)/.02;
end

yMax=max(max(psthNorm));
for i=1:8
    subplot(3,3,graph(i))
    bar(edgesPeri,psthNorm(:,i))
    ylim([0 yMax])
    xlim([-1.05 1.05])
end

h=subplot(3,3,5);
set(h,'Visible','off')
chanNum=unit(neuronNum).chanNum;
unitNum=unit(neuronNum).unitNum;
text(0.25,0.5,['Chan ' num2str(chanNum) '-' num2str(unitNum)])
    