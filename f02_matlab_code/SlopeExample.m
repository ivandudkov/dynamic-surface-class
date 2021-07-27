
clear variables
close all
load('Nubble_162020_sampleSwath.mat')
load('Nubble_DTMs.mat')

p=0; % Assume zero pitch

% there are still a lof of spikes in the data - filter them out (this
% really should be completely handled before, but that's ok for now

%MB1_dtm.FilterSD(9,3) % This is painfully slow... I already saved the result of this 
%                     % so you don't have to rerun it, but you're welcome to
%                     % do so!
bottom_dtm.FilterSD(27,3); % Same deal

screensize = get( groot, 'Screensize' );
fig=figure();
fig.Position=[0 screensize(4)/2 screensize(3)/2 screensize(4)/2];
subplot(1,2,1);
bottom_dtm.Plot(0) % Plots with outliers
subplot(1,2,2);
bottom_dtm.Plot(1) % Plots with outliers removed
                    
bot_copy=bottom_dtm; % Copy in which we're not too upset deleting data

bot_copy.Removedata(); %Delete data identified by the mask
                       % You will notice that bottom_dtm.Plot(0) and
                       % bottom_dtm.Plot(1) will now produce the same
                       % result

% For now only do three of the bottom picks of the swath - just for
% illustration

bot_copy.AofI(bottom_x(1:3),bottom_y(1:3),bottom_th(1:3),heading); 
                                            % Determine the angle of intersection - you 
                                            % will have to complete this
                                            % function, but I got it
                                            % started in such a way that it
                                            % hopefully answers your
                                            % question

                                            
% Note that your canopy dtm is not the same size as your bottom dtm, It
% would make life easier if they were - I'm not sure how you do the
% differencing between the two?

