classdef RegGrid3D  < matlab.mixin.Copyable & dlnode
    % This class hold a UNB style weighed grid - it really consists of two
    % grids, namely a grid of the sum all of the weights of all the
    % contrbutions to the grid cell and a corresponding grid with the sum
    % of all the weighed contributions weighed.
    
    % the big benefit is that you can easily remove and add contributions
    % to the grid
    % Note that the implicit understanding is that the units in the x and y
    % directions are isometric
    
    % Semme Dijkstra    Nov 23, 2016
    % Added AofI function               %2017 12 13     Semme Dijkstra
    % Added Filter function             %2017 12 13     Semme Dijkstra
    % Added mask to properties          %2017 12 13     Semme Dijkstra
    % Updated Plot function             %2017 12 13     Semme Dijkstra
    
    properties(GetAccess = 'public', SetAccess = 'private')
        X;          % regularly spaced array in X xirection
        Y;          % regularly spaced array in y direction
        rangeX;     % X range of the grid
        rangeY;     % Y range of the grid
        sumWeight;  % regularly spaced array of summed weights
        weighGrid;  % Grid value weighed by sumWeight
        res;
        rInfl;
        rInc;
        rPix;
        kWeight;    % Weigthing parameter for Kernel, usually inverse 
                    % distance squared in either pixels or meters
        mask;
    end
    
    methods
        function obj=RegGrid3D(res,rInfl,rInc,rangeX,rangeY)
            
            obj.rInc=0;
            obj.res=res;
            obj.rInfl=rInfl;
            obj.rPix=round(rInfl/res);
            if nargin>=3
                obj.rInc=rInc;
            end
            if nargin>=4
                obj.rangeX=rangeX;
                obj.rangeY=rangeY;
                
                % Create a new meshgrid covering the expanded range, Note
                % the way I set up the grid - this to avoid problems with
                % rounding errors
                % The original statement was the more commonly used
                % variant
                % [obj.X,obj.Y] = meshgrid(obj.rangeX(1):obj.res:obj.rangeX(2),obj.rangeY(1):obj.res:obj.rangeY(2));
                % But rounding errors smaller than e-16 could lead to one
                % whole pixel (row or column) rounding error - this was one
                % of the more difficult to diagnose bugs that I ever dealt
                % with...
                [obj.X,obj.Y] = meshgrid( ...
                    double((1:uint64(diff(obj.rangeX)/obj.res)))*obj.res+obj.rangeX(1),...
                    double((1:uint64(diff(obj.rangeY)/obj.res)))*obj.res+obj.rangeY(1));
                
                % Make sure that the stored ranges are an exact representation
                % of the meshgrid
                
                obj.rangeX(2)=obj.X(end,end);
                obj.rangeY(2)=obj.Y(end,end);
                [numRows,numCols]=size(obj.X);
                
                % Now create the weighed grid and the associated grid of
                % weights. Note that we should test for lack of data by 0s in
                % the weights, not in the weighed depths; zero is a meaningful
                % (and often important) value in grids
                
                obj.weighGrid=zeros(numRows,numCols);
                obj.sumWeight=obj.weighGrid;
                
                % Update the ranges so that they are representative of what
                % is in the grid
                
                obj.rangeX(1)=obj.X(1,1);
                obj.rangeX(2)=obj.X(1,end);
                obj.rangeY(1)=obj.Y(1,1);
                obj.rangeY(2)=obj.Y(end,1);
            else
                obj.rangeX=nan(2,1);
                obj.rangeY=nan(2,1);
            end
            
            
            % Create a distance weighting kernel that weighs by 1/R^2,
            % except for R=0, for which the weight = 1
            
            [wx,wy]=meshgrid(-obj.rPix:obj.rPix,-obj.rPix:obj.rPix);
            obj.kWeight=1./(wx.^2+wy.^2);
            
            % Deal with the weight at the center i.e., R=0
            obj.kWeight(obj.rPix+1,obj.rPix+1)=1;
            
            % The kernel just defined is a square, but the radius of
            % influence defines a circle - set all the weights outside
            % the circle to zero this to create circular footprints of
            % contributing data
            
            obj.kWeight(obj.kWeight<obj.rPix^-2)=0;
            
            % update the object''s radius of influence
            
            obj.rInfl=rInfl;
            
        end
        
        function Create(obj,x,y,z,obsWeight)
            
            % Determine the area of influence in number of pixels, assume
            % isometric coordinates
            
            
            
            % Create arrays that cover the full data extent,
            % expanded by the radius of influence in pixel units to
            % ensure that the all data can be fully captured in the arrays
            
            obj.rangeX=nan(2,1);
            obj.rangeY=nan(2,1);
            
            obj.rangeX(1)=min(x)-obj.rPix;
            obj.rangeX(2)=max(x)+obj.rPix;
            obj.rangeY(1)=min(y)-obj.rPix;
            obj.rangeY(2)=max(y)+obj.rPix;
            
            % Create a new meshgrid covering the expanded range, Note
            % the way I set up the grid - this to avoid problems with
            % rounding errors
            % The original statement was the more commonlus used
            % variant
            % [obj.X,obj.Y] = meshgrid(obj.rangeX(1):obj.res:obj.rangeX(2),obj.rangeY(1):obj.res:obj.rangeY(2));
            % But rounding errors smaller than e-16 could lead to one
            % whole pixel (row or column) rounding error - this was one
            % of the more difficult to diagnose bugs that I ever dealt
            % with...
            [obj.X,obj.Y] = meshgrid( ...
                double((1:uint64(diff(obj.rangeX)/obj.res)))*obj.res+obj.rangeX(1),...
                double((1:uint64(diff(obj.rangeY)/obj.res)))*obj.res+obj.rangeY(1));
            
            % Make sure that the stored ranges are an exact representation
            % of the meshgrid
            
            obj.rangeX(2)=obj.X(end,end);
            obj.rangeY(2)=obj.Y(end,end);
            [numRows,numCols]=size(obj.X);
            
            % Now create the weighed grid and the associated grid of
            % weights. Note that we should test for lack of data by 0s in
            % the weights, not in the weighed depths; zero is a meaningful
            % (and often important) value in grids
            
            obj.weighGrid=zeros(numRows,numCols);
            obj.sumWeight=obj.weighGrid;
            
            
            % Loop through the data - for now using a for loop
            
            
            for i=1:length(z)
                
                % Get the location of the data in the grid
                
                xGrid=find(x(i)>=obj.X(1,:),1,'last');
                yGrid=find(y(i)>=obj.Y(:,1),1,'last');
                
                % Set the location of associated kernel in the grid
                
                k=[xGrid-obj.rPix,xGrid+obj.rPix;yGrid-obj.rPix,yGrid+obj.rPix];
                
                % Add the contribution to both the weighed grid as well as
                % as the grid of summed weights
                
                obj.sumWeight(k(2,1):k(2,2),k(1,1):k(1,2))=...
                    obj.sumWeight(k(2,1):k(2,2),k(1,1):k(1,2))+...
                    obsWeight*obj.kWeight;
                
                obj.weighGrid(k(2,1):k(2,2),k(1,1):k(1,2))=...
                    obj.weighGrid(k(2,1):k(2,2),k(1,1):k(1,2))+...
                    obsWeight*obj.kWeight*z(i);
            end
            
            %  surf(X,Y,obj.weighGrid./obj.sumWeight);
        end
        
        
        % Add an array of observations
        
        function Add(obj,x,y,z,obsWeight)
            
            if isempty(x)||isempty(y)||isempty(z)
                return
            end
            
            % Make sure that arrays cover the full data extent,
            % expanded by the radius of influence in pixel units to
            % ensure that all data can be fully captured in the arrays
            
            if isnan(obj.rangeX(1))
                %There is no data yet - define the dimension of the grid so
                %that all data can be held
                
                obj.rangeX(1)=min(x)-obj.rPix;
                obj.rangeX(2)=max(x)+obj.rPix;
                obj.rangeY(1)=min(y)-obj.rPix;
                obj.rangeY(2)=max(y)+obj.rPix;
                
                % Create a new meshgrid covering the expanded range, Note
                % the way I set up the grid - this to avoid problems with
                % rounding errors
                % The original statement was the more commonly used
                % variant
                % [obj.X,obj.Y] = meshgrid(obj.rangeX(1):obj.res:obj.rangeX(2),obj.rangeY(1):obj.res:obj.rangeY(2));
                % But rounding errors smaller than e-16 could lead to one
                % whole pixel (row or column) rounding error - this was one
                % of the more difficult to diagnose bugs that I ever dealt
                % with...
                [obj.X,obj.Y] = meshgrid( ...
                    double((1:uint64(diff(obj.rangeX)/obj.res)))*obj.res+obj.rangeX(1),...
                    double((1:uint64(diff(obj.rangeY)/obj.res)))*obj.res+obj.rangeY(1));
                
                % Make sure that the stored ranges are an exact representation
                % of the meshgrid
                
                obj.rangeX(2)=obj.X(end,end);
                obj.rangeY(2)=obj.Y(end,end);
                [numRows,numCols]=size(obj.X);
                
                % Now create the weighed grid and the associated grid of
                % weights. Note that we should test for lack of data by 0s in
                % the weights, not in the weighed depths; zero is a meaningful
                % (and often important) value in grids
                
                obj.weighGrid=zeros(numRows,numCols);
                obj.sumWeight=obj.weighGrid;
                
                
                
            else
                cMinX=0;
                cMaxX=0;
                cMinY=0;
                cMaxY=0;
                
                % Determine by how many pixels the grid should be increased
                % in each direction
                
                if min(x)-obj.rPix < obj.rangeX(1)
                    if ~obj.rInc
                        cMinX=double(ceil((obj.rangeX(1)-(min(x)-obj.rPix))/obj.res));
                    else
                        cMinX=obj.rInc;
                    end
                    obj.rangeX(1)=obj.rangeX(1)-cMinX*obj.res;
                    
                end
                if max(x)+obj.rPix > obj.rangeX(2)
                    if ~obj.rInc
                        cMaxX=double(ceil((max(x)+obj.rPix-obj.rangeX(2))/obj.res));
                    else
                        cMaxX=obj.rInc;
                    end
                    obj.rangeX(2)=obj.rangeX(2)+double(cMaxX)*obj.res;
                end
                if min(y)-obj.rPix < obj.rangeY(1)
                    if ~obj.rInc
                        cMinY=double(ceil((obj.rangeY(1)-(min(y)-obj.rPix))/obj.res));
                    else
                        cMinY=obj.rInc;
                    end
                    obj.rangeY(1)=obj.rangeY(1)-cMinY*obj.res;
                end
                if max(y)+obj.rPix > obj.rangeY(2)
                    if ~obj.rInc
                        cMaxY=double(ceil((max(y)+obj.rPix-obj.rangeY(2))/obj.res));
                    else
                        cMaxY=obj.rInc;
                    end
                    obj.rangeY(2)=obj.rangeY(2)+cMaxY*obj.res;
                end
                
                if cMinX||cMinY||cMaxX||cMaxY
                    
                    % Get a copy of the existing grid
                    
                    wg=obj.weighGrid;
                    sw=obj.sumWeight;
                    
                    
                    
                    % Create a new meshgrid covering the expanded range, Note
                    % the way I set up the grid - this to avoid problems with
                    % rounding errors
                    [obj.X,obj.Y] = meshgrid( ...
                        double((1:uint64(diff(obj.rangeX)/obj.res)))*obj.res+obj.rangeX(1),...
                        double((1:uint64(diff(obj.rangeY)/obj.res)))*obj.res+obj.rangeY(1));
                    % Make sure that the stored ranges are an exact representation
                    % of the meshgrid
                    obj.rangeX(2)=obj.X(end,end);
                    obj.rangeY(2)=obj.Y(end,end);
                    
                    [numRows,numCols]=size(obj.X);
                    
                    % Create the new weighed grid and associated sum weights
                    
                    obj.weighGrid=zeros(numRows,numCols);
                    obj.sumWeight=obj.weighGrid;
                    
                    % Preserve the previously determined values
                    
                    obj.weighGrid(1+cMinY:end-cMaxY,1+cMinX:end-cMaxX)=wg;
                    obj.sumWeight(1+cMinY:end-cMaxY,1+cMinX:end-cMaxX)=sw;
                end
            end
            
            
            % Loop through the data - for now using a for loop
            
            
            for i=1:length(z)
                
                % Get the location of the data in the grid
                
                xGrid=find(x(i)>=obj.X(1,:),1,'last');
                yGrid=find(y(i)>=obj.Y(:,1),1,'last');
                
                % Set the location of associated kernel in the grid
                
                k=[xGrid-obj.rPix,xGrid+obj.rPix;yGrid-obj.rPix,yGrid+obj.rPix];
                
                % Add the contribution to both the weighed grid as well as
                % as the grid of summed weights - note that the xindex is
                % the columns and the y index is the rows and that rows
                % come before columns
                
                obj.sumWeight(k(2,1):k(2,2),k(1,1):k(1,2))=...
                    obj.sumWeight(k(2,1):k(2,2),k(1,1):k(1,2))+...
                    obsWeight*obj.kWeight;
                
                obj.weighGrid(k(2,1):k(2,2),k(1,1):k(1,2))=...
                    obj.weighGrid(k(2,1):k(2,2),k(1,1):k(1,2))+...
                    obsWeight*obj.kWeight*z(i);
            end
        end
        
        function aoi=AofI(obj,x,y,bh,h)
            % x is Easting
            % y is Northing
            % bh is depression angle
            
            aoi=nan;
            if isempty(x)||isempty(y)
                return;
            end
            
            nS=length(x);
            
            if length(y)~=nS
                error('Vectors x and y must match dimensions!')
            end
            
            % Determine whether the current location is covered by the
            % grid
            
            if isnan(obj.rangeX(1))
                return;
            else
                if (min(x) <= obj.rangeX(1))|| ...
                        (max(x) >= obj.rangeX(2))|| ...
                        (min(y) <= obj.rangeY(1))|| ...
                        (max(y) >= obj.rangeY(2))
                    return;
                end
            end
            
            [nRows,~]=size(obj.X);
            
            % Loop through the data - for now using a for loop
                      
            for i=1:nS
                
                % Get the location of the data in the grid
                
                xGrid=find(x(i)>=obj.X(1,:),1,'last');
                yGrid=find(y(i)>=obj.Y(:,1),1,'last');
                
                % xGrid is now your column, yGrid is your row
                
                disp(['xGrid: ' num2str(xGrid)]);
                disp(['yGrid: ' num2str(yGrid)]);
                
                % to demonstrate the indexing look at the value from this
                % location
                
                d_two=obj.weighGrid(yGrid,xGrid)./obj.sumWeight(yGrid,xGrid);
                
                % Now look at one dimensional indexing of the same grid
                
                ii=yGrid+(xGrid-1)*nRows;
                
                d_one=obj.weighGrid(ii)./obj.sumWeight(ii);
                
                disp(['Depth using 2d indexing: ' num2str(d_two)])
                disp(['Depth using 1d indexing: ' num2str(d_one)])
                
                % So ii is the index of the vertex on the lower left of the
                % pixel in which the swath intersects the bottom i.e. the
                % pixels is surrounded by the vertices
                % LL=(yGrid,xGrid)
                % LR=(yGrid,xGrid+1)
                % UL=(yGrid+1,xGrid)
                % UR=(yGrid+1,xGrid)
                
                % Calculate the direction vector and normalize it, also
                % offset the heading by 90 degrees
                
                h=h+pi/2;
                u=sqrt([cos(h)^2*(1-sin(bh(i))^2) sin(h)^2*(1-sin(bh(i))^2) sin(bh(i))^2]);
                    
                % Note you have to careful about the signs
                % I did not check this for you
                % tan(asin(u(1,3)))*abs(cross distance) should get you
                % approximately to the depth observed
                
                % etc
            end
            
        end
        
        function FilterSD(obj,Size,crit,varargin)
            
            if ~rem(Size,2)||(Size<1)||(floor(Size)~=Size)
                error('Kernel size must be an uneven integer greater than 1')
            end
            
            % VERY simple minded approach to get rid of the most significant
            % spikes - note that it does not deal with the edges - update
            % the algorithm to do that - this is very SLOW, but works
            
            nNan=Size^2/2+1;
            Size=floor(Size/2);
            
            % Create a sizexsize normalized filter
            
            [m,n]=size(obj.X);
            
            dtm=obj.weighGrid./obj.sumWeight; 
            
            dtm_m=nan(m,n);
            dtm_sd=nan(m,n);
            
            % By default mask all data 
            
            obj.mask=true(m*n,1);

            for i=Size+1:m-Size
                for j=Size+1:n-Size
                    r=i-Size:i+Size;
                    c=j-Size:j+Size;
                    k=dtm(r,c);

                    % If there are too many nan's don't bother calculating
                    % This has the risk of masking all the borders so allow
                    % up to half the numbers+1 to be nan's
                     
                    if sum(~isnan(k(:)))>nNan
                            obj.mask(i+(j-1)*m)=false;  %unmask the data here
                            dtm_m(i,j)=mean(k(:),'omitnan');
                            dtm_sd(i,j)=std(k(:),'omitnan');
                     end
                end
            end
            
            % The mean standard deviation
            
            m_sd=mean(dtm_sd(:),'omitnan');
            
            % Also mask the locations where the standard deviation gets too out
            % of hand i.e. the really big isolated spikes
            
            obj.mask=obj.mask|dtm_sd(:)> 3*crit*m_sd; 
            
            % Finally,mask the locations where the std is greater than crit times
            % the mean standard deviation
            
            obj.mask=obj.mask|abs(dtm_m(:)-dtm(:))>crit*dtm_sd(:);
            
            % If we chose to replace values then do it using the mask
            
            if nargin==4
               if varargin{1} 
                    obj.weighGrid(obj.mask)=dtm_sd(obj.mask);
                    obj.sumWeight=1;
               end
            end
        end
        
        function Removedata(obj)
            % Remove all data covered by the mask
            
            obj.weighGrid(obj.mask)=nan;
            obj.sumWeight(obj.mask)=nan;
        end
        
%         function FilterDiff(obj,dtm,crit)
%             % Filter by a difference with another terrain model
%             % if the difference is too big delete the data
%             % This allows you to easily filter crazy canopy heights
%             
%             [m,n]=size(obj.X);
%             
%             if ~all([m,n]==size(dtm.X))
%                 error('Grids must match in size!')
%             end
%             
%             % Determine the mean difference
%             
%             % Stopped here as the canopy and bottom dtm's are different
%             % sizes - you could calculate the bottom dtm and then the
%             % canopy dtm - I'm not sure how you do the differencing?
%             
%         end
        
        function Plot(obj,mask,varargin)
            
%             if nargin==3
%                 h=varargin{1};
%             end
            
            % This is certainly not the fastest way to plot the data (that
            % would be using the surface functions) but it does allow for
            % easier manipulation of the data - note that adding lighting
            % to this would be nice and slick - this will not be hard to do
            exag=20;

            dtm=obj.weighGrid./obj.sumWeight*exag;
            if mask
                dtm(obj.mask)=nan;
            end
            surf(dtm,'LineStyle','none')
            i=xticklabels;
            i=cellfun(@str2num,i);
            i=obj.X(1)+i*obj.res;
            xticklabels({num2str(i,'%.2f')})
            xlabel('Easting (m)');
            i=yticklabels;
            i=cellfun(@str2num,i);
            i=obj.Y(1)+i*obj.res;
            yticklabels({num2str(i,'%.2f')});
            ylabel('Northing (m)');
            i=zticklabels;
            i=cellfun(@str2num,i);
            i=i/exag;
            zticklabels({num2str(i,'%.2f')});
            zlabel('Depth (m)');
            if mask
                title('DTM: Masked Bathymetry');
            else
                title('DTM: Non Masked Bathymetry');
            end
            c=colorbar('vert');
            c.TickLabels={num2str(i,'%.2f')};
            axis equal
            view(150,23);
        end
    end
    
end

