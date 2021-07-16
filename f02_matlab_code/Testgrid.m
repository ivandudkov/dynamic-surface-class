%% Testgrid

clear all
close all

% Create arrays of x and y cooordinates

[x,y]=meshgrid(1:100);

% Calculate z values

z=sin((x+y)*pi/180);

figure
subplot(2,2,1);
surf(z);

% Now create a similar model, but on a higher resolution grid, with only a
% z value for every 10 the row and column

z=nan(100);

for i=2:10:100
    for j=2:10:100
        z(j-1,i-1)=sin((x(j-1,i-1)+y(j-1,i-1))*pi/180);
        z(j  ,i-1)=sin((x(j  ,i-1)+y(j  ,i-1))*pi/180);
        z(j-1,i  )=sin((x(j-1,i  )+y(j-1,i  ))*pi/180);
        z(j  ,i  )=sin((x(j  ,i  )+y(j  ,i  ))*pi/180);
    end
end
subplot(2,2,2);
surf(z);

% Get the coordinates of where data is available

[row,col]=find(~isnan(z));
Z=z(row,col);

% Now calculate a DEM based on the input data from x, y and z

x=[0 500 1000];
y=[0 500 2000];
z=[1 4 1];

dem=RegGrid3D(3,20,20,[min(x) max(x)],[min(y) max(y)]);
dem.Create(x,y,z,1,1,5);

dem.Plot();
surface=dem.weighGrid./dem.sumWeight;
surface(surface==inf)=NaN;

subplot(2,2,3);
figure
surf(surface);

drawnow()