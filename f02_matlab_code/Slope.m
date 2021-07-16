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

% filter span - guarantee an odd number and maximize at heading 45
span=0;
if sin(heading)~=0
    span=2*(floor(span/sin(heading)+1))+1;
else
    span=1;
end

span=3;
% Visualiszation parameters

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

aspect=atan2(Nx,Ny);

% Map it to from [-pi,pi] to [0,2pi] range (just so that we are consistent
% with the normal range of azimuths)

aspect(aspect(:)<0)=aspect(aspect(:)<0)+2*pi;


%% Generate a swath and intersect it with the bottom
% Note that for convenience I'll just calculate a bunch of rays

crossDist=(0:1:sqrt(2*(vertices/2)^2))';

% Make the cross distance range from port to starboard

crossDist=[wrev(-crossDist);crossDist(2:end)];

% Identify all the pixels illuminated by the swath 

swath=crossDist.*[sin(heading+pi/2) cos(heading+pi/2)];

% Filter out the rays that extend outside the DEM

de=abs(swath(:,1))>vertices/2|abs(swath(:,2))>vertices/2;
swath(de,:)=[];
crossDist(de,:)=[];

% Identify the pixels in the DEM that are intersected by the swath (note
% that if the pixels are not 1x1 you need to do some scaling

index=floor(swath+(vertices-1)/2);
jndex=swath+(vertices-1)/2;
iLL=(index(:,1)-1)*vertices+index(:,2);
iLR=(index(:,1))*vertices+index(:,2);
iUL=(index(:,1)-1)*vertices+index(:,2)+1;
iUR=(index(:,1))*vertices+index(:,2)+1;

de=iLL<0|iLR<0|iUL<0|iUR<0;
swath(de,:)=[];
crossDist(de,:)=[];
index(de)=[];
jndex(de)=[];
iLR(de)=[];
iUL(de)=[];
iUR(de)=[];
iLL(de)=[];

% Calculate the depth by linear interpolation (technically speaking you
% should not do this, but rather intersect the pixel plane with the ray,
% however since we have one pixel representing the footprint we may as well
% not bother - note that due to the way the terrain model is calculated,
% taking into the account the footprint, this pixel should indeed be a good
% estimate of the average slope of the entire footprint.

% Start linear interpolation at the bottom of the pixels in x-dir
dL=(dtm(iLR)-dtm(iLL)).*(jndex(:,1)-index(:,1))+dtm(iLL);
% continue linear interpolation at the top of the pixels in x-dir
dU=(dtm(iUR)-dtm(iUL)).*(jndex(:,1)-index(:,1))+dtm(iUL);
% Finish by interpolation in the y-dir
swath=[swath (dU-dL).*(jndex(:,2)-index(:,2))+dL];

% note the quantization like effects that may appear (try heading 45) that
% results from the pixelation of the DTM - this will lead to a direction
% dependency in the accuracy of the slope. If the DTM is aligned NS, EW
% then the errors in the cardinal directions are minimized- You can see the
% effect occuring in the Slope profile plots

% Just make the swath complete by including the cross distances

swath=[swath crossDist];



%% Alternative A: Intersect the swath with bottom at beams boresights 
% determine interaction length from slope - this is what you propose to do
% based on the paper

% Determine the grazing angle wrt to the horizontal - this is the same as 
% the depression angle at the end of your ray tracing and you should
% already have this 

% There are two ways in which you can calculate this

%% Method 1 By creation of a bottom profile and calculating the directional
% derivative along it

% Determine the orthogonal vectors - NOTE that we have positive going down - in
% other words the polarity of the resulting slope is swapped from what you are used to 

slopeProf=[diff(swath(:,3)),diff(swath(:,4))];

% Normalize the orthogonal vectors

slopeProf=slopeProf./(sqrt(sum(slopeProf.^2,2)));

% These are the normals at the halway point of the vertices move them to
% the vertices by averaging - note that we are taking an arithmetic mean 
% where where we really should be using a harmonic mean (but its good enough)
% - also note how I dealt with the end members - in the graph you will see 
% that this is a much better representation and not as sensitive to quantization

nProf=[slopeProf(1,:);slopeProf];
nProf(1:end-1,:)=(nProf(1:end-1,:)+nProf(2:end,:))/2;
nProf(end,:)=nProf(end-1,:);

% The slopes in the direction of the swaths are then

slope1=asin(nProf(:,1))*180/pi;

% We may still not be quite there - we are taking derivatives at boundaries,
% which may create some weirdness - you can smooth by a moving average
% THIS MAY AFFECT THE OUTCOMES AND WE DO NOT WANT TO OVERFILTER - this inmo
% itself could be a point of study. - Try with and without filtering and
% see what happens

slope1=smooth(slope1,span);

% Now if you try this at a heading of 45 you will see that there are all
% kinds of numerical issues - because of the alignment of the DTM with the swath
% Using a different depth determination method will lessen this BUT there
% is also just msmt noise dependency (not in this simulated case) which
% requires filtering. Low pass filtering rather than a simple moving average
% should give better results

% Remember that you will need to correct to a standard depth!!!!!

%% Method 2 - look at the surface normals provided by the surfnorm function

% Start linear interpolation at the bottom of the pixels in x-dir
NxLo=(Nx(iLR)-Nx(iLL)).*(jndex(:,1)-index(:,1))+Nx(iLL);
NyLo=(Ny(iLR)-Ny(iLL)).*(jndex(:,1)-index(:,1))+Ny(iLL);
NzLo=(Nz(iLR)-Nz(iLL)).*(jndex(:,1)-index(:,1))+Nz(iLL);
% continue linear interpolation at the top of the pixels in x-dir
NxUp=(Nx(iUR)-Nx(iUL)).*(jndex(:,1)-index(:,1))+Nx(iUL);
NyUp=(Ny(iUR)-Ny(iUL)).*(jndex(:,1)-index(:,1))+Ny(iUL);
NzUp=(Nz(iUR)-Nz(iUL)).*(jndex(:,1)-index(:,1))+Nz(iUL);
% Finish by interpolation in the y-dir
NxLe=(NxUp-NxLo).*(jndex(:,2)-index(:,2))+NxLo;
NyLe=(NyUp-NyLo).*(jndex(:,2)-index(:,2))+NyLo;
NzLe=(NzUp-NzLo).*(jndex(:,2)-index(:,2))+NzLo;

% For clarity's sake let's map the swath vectors to unity vectors u
% First let's look at the slope - since this is what you get from method
% one - it is simply the dot product of the direction vector of the swath
% and the surface normals (but the sines, since we are dealing with the 
% normals rather than the gradient)

u2=[swath(1,1:2)./sqrt(sum(swath(1,1:2).^2,2)) 0];
slope2=zeros(length(swath),0);

for i=1:length(swath)
    slope2(i)=asin(u2*[NxLe(i);NyLe(i);NzLe(1)])*180/pi;
end

slope2=smooth(slope2,span);

%% Compound grazing angle
% We can also determine the compound grazing angle directly using the normalized 
% direction vectors of the swath

u=swath(:,1:3)./sqrt(sum(swath(:,1:3).^2,2));
grazing=zeros(length(swath),0);

% The compound grazing angles are then simply the result of the dot products

for i=1:length(swath)
    grazing(i)=asin(u(i,:)*[NxLe(i);NyLe(i);NzLe(1)])*180/pi;
end

% Note that close to nadir we may get some extremes leading to imaginary
% numbers - just take the real part (this is because we used linear
% interpolation rather than intersecting a vector with a plane)

grazing=real(grazing);

% To be consistent we'll smooth this as well
grazing=smooth(grazing,span);





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
fhd=figure('Name','Slope Example - DEM data, Semme Dijkstra', ...
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

fhp=figure('Name','Slope Example - Profile Data, Semme Dijkstra', ...
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
for i=1:length(swath)-1
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
for i=1:length(swath)
    x=swath(i,4);
    y=swath(i,3);
    plot([x;x+nProf(i,1)*5],[y;y-nProf(i,2)*5])
end
hold off
axis equal
set(ax2_4,'Ydir','reverse')

ax2_5=subplot(3,2,5);
plot(swath(:,4),slope1);
title('Slope Method 1 - problems at heading 45 mod 90');
hold on
hold off

ax2_6=subplot(3,2,6);
plot(swath(:,4),slope2);
title('Slope Method 2: Note how much less problematic at heading 45 mod 90');
hold on
hold off

figure
plot(swath(:,4),grazing);
title('Grazing from Method 2 is easy');

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

