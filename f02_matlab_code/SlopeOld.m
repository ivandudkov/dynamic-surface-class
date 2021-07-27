%% Slope

% This script illustrates some slope calculations - this file written for
% Ashley Norton to help with slope compensation
%
% Semme J. Dijkstra Oct 3, 2017

% Based on some Matlab examples

close all
clear variables
clc

%% Parameter Initialization

% Generate a DTM with an odd number of vertices with an average depth of ~20 m
% Note that the horizontal distance unit is meters and that we will create
% a 1x1 m pixelsize

vertices=101;           % Number of vertices
aveD=20;                % Approx. Average depth

dtm=peaks(vertices)+aveD;

% Swath parameters

position=[vertices/2 vertices/2];
heading=45*pi/180;      % Heading of Tx
Tx=[0,0,1];             % position of Tx
SwathWidth=140*pi/180;  % Swath width
numBeams=100;    

% Visualization parameters

ve=7;                   % Vertcal Exageration;


%% Generate derivative grids

% Determine the surface normal components

[Nx,Ny,Nz]=surfnorm(dtm);
figure; surfnorm(dtm);
axis equal

% The vectors [Nx(i,j) Nx(i,j) Nx(i,j)] are normalized i.e., the components
% are the direction cosines - thus the Nz component is the cosine of the surface
% slope i.e., the 'up' gradient

slope=acos(Nz);

% Now that we know the slope we can also determine the direction of the
% slope by determining the aspect. To get in the right quadrant we will use
% the atan2 function

aspect=atan2(Ny,Nx);

% Map it to from [-pi,pi] to [0,2pi] range (just so that we are consistent
% with the normal range of azimuths)

aspect(aspect(:)<0)=aspect(aspect(:)<0)+2*pi;


%% Generate a swath and intersect it with the bottom
% Note that for convenience I'll just calculate a ray to each of the pixels
% along the swath. You will know the array intersects the bottom from the
% ray tracing - You will also know the angles from the heading and the ray
% tracing - most of the code in this section is steps that you do not need,
% but it is important that you understand the thinking behind it

crossDist=(0:1:sqrt(2*(vertices/2)^2))';

% Identify all the pixels illuminated by the swath on port; increase
% the range by 1 through crossDistMax to ensure that all pixels are covered
% Note the use of the wrev function to flip the cross distance vector so
% that we end 

swath=wrev(crossDist).*[sin(heading-pi/2) cos(heading-pi/2)];

% Now do the port - starting from nadir, so no wrev needed.

swath=[swath;crossDist.*[sin(heading+pi/2) cos(heading+pi/2)]];

% Map the swath to integer vertices

swath=round(swath);

% Filter out the rays that extend outside the DEM

swath(abs(swath(:,1))>(vertices-1)/2|abs(swath(:,2))>(vertices-1)/2,:)=[];

% Map the distances to indexes in the dtm and other related matrices

index=swath+(vertices-1)/2;

% Filter out any double entries - need to do this otherwise we may get odd
% artifacts in the profile (see what happens if you comment out the
% deletion statements - very obvious at heading 45 deg

dblEnt=diff(swath);
dblEnt=[false;~dblEnt(:,1)&~dblEnt(:,2)];
swath(dblEnt,:)=[];
index(dblEnt,:)=[];

% Identify the pixels in the DEM that are intersected by the swath, but as
% if the dtm is a one dimensional array

index=(index(:,1)-1)*vertices+index(:,2);

% Filter out any indexes that are outside of the array

swath(index<0,:)=[];
index(index<0)=[];

% Create the vectors from the Tx to the intersections - we already know the
% x and y components, but not yet the depths. The easiest way of doing this
% is by adressing the dtm as a linear array at the indexes specified by the
% swath. Let's round the swath x and y offsets towards zero to be more
% consistent with the indexes

swath=[swath dtm(index)];

% note the quantization like effects that may appear (try heading 45) that
% results from the pixelation of the DTM - this will lead to a direction
% dependency in the accuracy of the slope. If the DTM is aligned NS, EW
% then the errors in the cardinal directions are minimized- You can see the
% effect occuring in the Bathymetry profile plot - Note that the slope is
% not determined from analysis of the bathymetry profile, but rather from
% the already existing set of normalized vectors

% To determine the slope along the profile we need to calculate the 
% tangents along it, to do this accurately we need to have the correct 
% distances between the points - let's do by creating a distance scale
% centered on the transducer

swath=[swath [sqrt(sum(swath(1:vertices/2-.5,1:2).^2,2));-sqrt(sum(swath((vertices+1)/2:end,1:2).^2,2))]];



%% Alternative A: Intersect the swath with bottom at beams boresights 
% determine interaction length from slope - this is what you propose to do
% based on the paper

% Determine the grazing angle wrt to the horizontal - this is the same as 
% the depression angle at the end of your ray tracing and you should
% already have this 

% There are two ways in which you can calculate this

% 1 By creation of a bottom profile and calculating the directional
% derivative along it

% Determine the normal vectors - NOTE that we have positive going down - in
% other words the polarity of the resulting slope is swapped from what you are used to 

slopeProf=[diff(swath(:,3)),diff(swath(:,4))];

% % Pad the end to avoid weirdness at the boundary of the DTM - assume that the last
% % slope is the same as the one before it
% 
% slopeProf=[slopeProf;slopeProf(end,:)];

% Normalize the normal vectors

slopeProf=slopeProf./(sqrt(sum(slopeProf.^2,2)));

% These are the normals at the halway point of the vertices move them to
% the vertices by averaging - note that we are taking an arithmetic mean 
% where where we really should be using a harmonic mean (but its good enough)
% - also note how I dealt with the end members - in the graph you will see 
% that this is a much better representation and not as sensitive to quantization

nProf=[slopeProf(1,:);slopeProf];
nProf(1:end-1,:)=(nProf(1:end-1,:)+nProf(2:end,:))/2;
nProf(end,:)=nProf(end-1,:);

% We may still not be quite there - we are taking derivatives at boundaries,
% which may create some weirdness - you can smooth by a moving average
% THIS MAY AFFECT THE OUTCOMES AND WE DO NOT WANT TO OVERFILTER - this inmo
% itself could be a point of study. - Try with and without filtering and
% see what happens

nProf(:,1)=smooth(nProf(:,1)); % Zero phase 5 point moving average
nProf(:,2)=smooth(nProf(:,2)); % Zero phase 5 point moving average

% The slopes in the direction of the swaths are then

slope1=asin(nProf(:,1))*180/pi;

% Now if you try this at a heading of 45 you will see that there are all
% kinds of numerical issues


% Remember that you will need to correct to a standard depth!!!!!

% Method 2 - look at the surface normals provided by the surfnorm function

% slope2=acos(smooth(Nz(index)))*180/pi;
slope2=flipud(dtm(index));


%% Alternative B: Intersect the inner and outer maxima of the beam with the 
% seafloor - I only list this here so that you know that this would be
% another way of getting the slope difference - this is actually how I
% would do it, but I understand that you would want to be able to put in a
% reference for the method that you use

%% Alternative C: Comprehensive intersection of entire beam with the seafloor
% This comes at high computing cost, but creates the best result short of
% convolving the entite beam pattern with the bottom - this is where we
% want to eventually go - this is however best left to another thesis,
% maybe even at the PhD level


%% Show the results

% Position the output figure window for the dtm data

scrnSize=get(groot,'screenSize');
fh=figure('Name','Slope Example - DEM data, Semme Dijkstra', ...
    'Position',[ 1 scrnSize(4)*2/16  scrnSize(3)/2 scrnSize(4)/1.5], ...
    'NumberTitle','off');

% Create the various subplots

ax1_1=subplot(3,2,1);
surf(dtm,'LineStyle','none')
title('DTM: Bathymetry');
colorbar('vert');
axis equal
set(ax1_1,'Zdir','reverse')

ax1_2=subplot(3,2,2);
surf(dtm,'LineStyle','none')
title('DTM: Bathymetry');
colorbar('vert');
view(0,90)
axis equal
set(ax1_2,'Zdir','reverse')

ax1_3=subplot(3,2,3);
surf(Nx*180/pi,'LineStyle','none')
title('DTM: E-W Slope (Deflection)');
colorbar('vert');
view(0,90)
axis equal

ax1_4=subplot(3,2,4);
surf(Ny*180/pi,'LineStyle','none')
title('DTM: N-S Slope (Deflection)');
colorbar('vert');
view(0,90)
axis equal

ax1_5=subplot(3,2,5);
surf(slope*180/pi,'LineStyle','none')
title('DTM: Slope');
colorbar('vert');
view(0,90)
axis equal

ax1_6=subplot(3,2,6);
surf(aspect*180/pi,'LineStyle','none')
title('DTM: Aspect (Direction of slope)');
colorbar('vert');
view(0,90)
axis equal


% Position the output figure window for the dtm data

fh=figure('Name','Slope Example - Profile Data, Semme Dijkstra', ...
    'Position',[ 1 scrnSize(4)*2/16  scrnSize(3)/2 scrnSize(4)/1.5], ...
    'NumberTitle','off');


% Create the various subplots

ax2_1=subplot(3,2,1);
surf(dtm,'LineStyle','none')
hold on
title('DTM: Bathymetry with Profile');
colorbar('vert');
axis equal
set(ax2_1,'Zdir','reverse')

plot3(swath(:,1)+(vertices-1)/2,swath(:,2)+(vertices-1)/2,swath(:,3),...
    'k','LineWidth',2);
hold off

ax2_2=subplot(3,2,2);
surf(dtm,'LineStyle','none')
hold on
title('DTM: Bathymetry with Profile');
colorbar('vert');
view(0,90)
axis equal
set(ax2_2,'Zdir','reverse')

plot3(swath(:,1)+(vertices-1)/2,swath(:,2)+(vertices-1)/2,swath(:,3),...
    'k','LineWidth',2);
hold off

ax2_3=subplot(3,2,3);
area(swath(:,4),swath(:,3));
title('DTM: Bathymetry Profile with Normals');
hold on
for i=1:length(index)-1
    % Note that the normals are defined for the point halfway between the
    % vertices!
    x=mean(swath(i:i+1,4));
    y=mean(swath(i:i+1,3));
    plot([x;x+slopeProf(i,1)*5],[y;y-slopeProf(i,2)*5])
end
hold off
axis equal
set(ax2_3,'Ydir','reverse')

ax2_4=subplot(3,2,4);
area(swath(:,4),swath(:,3));
title('DTM: Bathymetry Profile with Mean Normals at Vertices');
hold on
for i=1:length(index)
    % Note that the normals are defined for the point halfway between the
    % vertices!
    x=swath(i,4);
    y=swath(i,3);
    plot([x;x+nProf(i,1)*5],[y;y-nProf(i,2)*5])
end
hold off
axis equal
set(ax2_4,'Ydir','reverse')

ax2_5=subplot(3,2,5);
plot(swath(:,4),slope1);
title('Slope Method 1 - fails at Heading=40 mod 90');
hold on
hold off
axis equal

ax2_6=subplot(3,2,6);
% area(swath(:,4),swath(:,3));
plot(swath(:,4),slope2);
title('Slope Method 2');
hold on
hold off
axis equal
set(ax2_6,'Ydir','reverse')



%% Set the color maps

% Bathymetry
colormap (ax1_1,flipud(jet(64)));
colormap (ax1_2,flipud(jet(64)));
colormap (ax2_1,flipud(jet(64)));
colormap (ax2_2,flipud(jet(64)));

% Angles (we want a wrapping mapping so that we do not get a discontinuity 
% at angles leading up to 360 and away from 0

hmap(1:256,1) = linspace(0,1,256); 
hmap(:,[2 3]) = 0.7; %brightness 
huemap = hsv2rgb(hmap); 
colormap(ax1_3,huemap)
colormap(ax1_4,huemap)
colormap(ax1_5,jet(64))
colormap(ax1_6,huemap)

