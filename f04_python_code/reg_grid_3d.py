# For computations and plotting
import numpy as np
import matplotlib.pyplot as plt


# For hillshade:
# from matplotlib import cm
# from matplotlib.colors import LightSource


class RegGrid3D:
    """
    Original Matlab Class description:
    % This class hold a UNB style weighed grid - it really consists of two
    % grids, namely a grid of the sum all of the weights of all the
    % contributions to the grid cell and a corresponding grid with the sum
    % of all the weighed contributions weighed.

    % the big benefit is that you can easily remove and add contributions
    % to the grid
    % Note that the implicit understanding is that the units in the x and y
    % directions are isometric

    % Semme Dijkstra    Nov 23, 2016


    # Conversion to Python started on July 26, 2021
    # Conversion to Python finished on August 02, 2021 (first completed draft)
    # Functions "create" and "add" were debugged and tested on 09.08.2021
    # Ivan Dudkov
    """

    # Class constructor method
    def __init__(self, res, rinfl):
        print("Initialization of the RegGrid3D class\n")

        # Properties
        self.res = res  # Resolution
        self.rinfl = np.int64(rinfl)  # Radius of influence
        self.rpix = round(rinfl / res)  # Radius of influence in pixel units

        # Radius of influence in coordinate units
        self.rpix2coord = np.sum(np.arange(0, self.rpix + 1) * self.res)

        # Allocation memory for the ranges
        self.rangeX = np.zeros(2)
        self.rangeY = np.zeros(2)

        # Other variables
        self.X = None  # X-coord
        self.Y = None  # Y-coord
        self.weighGrid = None  # Weights Grid
        self.sumWeight = None  # Grid of the sum Weights
        self.dtmGrid = None  # Digital Terrain Model Grid
        self.stdGrid = None  # Standard Deviation grid
        self.std_sumWeight = None  # Grid of the sum Weights for std grid calculation
        self.std_weighGrid = None  # Weights Grid for std grid calculation

        # Create a distance weighting kernel that weighs by 1/R**2,
        # except for R=0, for which the weight = 1
        wx, wy = np.meshgrid(np.arange(-self.rpix, self.rpix + 1),
                             np.arange(-self.rpix, self.rpix + 1))

        # Create a kernel weighs grid
        np.seterr(divide='ignore')  # ignore zero-division warn
        self.kWeight = 1 / (wx ** 2 + wy ** 2)
        np.seterr(divide='warn')  # set back to default
        # Deal with the weight at the center i.e., R = 0
        self.kWeight[self.rpix, self.rpix] = 1
        self.kWeight = \
            np.where(self.kWeight < self.rpix ** -2, 0, self.kWeight)

    def create(self, x, y, z, obs_weight=1):
        # Original description from Matlab Script
        #         % Determine the area of influence in number of pixels,
        #         % assume isometric coordinates
        #         % Create arrays that cover the full data extent,
        #         % expanded by the radius of influence in pixel units to
        #         % ensure that the all data can be fully captured in the arrays

        self.rangeX[0] = np.amin(x) - self.rpix2coord
        self.rangeX[1] = np.amax(x) + self.rpix2coord
        self.rangeY[0] = np.amin(y) - self.rpix2coord
        self.rangeY[1] = np.amax(y) + self.rpix2coord

        # Create a new meshgrid covering the expanded range
        print("Creating the meshgrid using range vectors\n")

        self.X, self.Y = \
            np.meshgrid(
                np.arange(0, (np.ceil(np.diff(self.rangeX)[0] / self.res)) + 1)
                * self.res + self.rangeX[0],
                np.arange(0, (np.ceil(np.diff(self.rangeY)[0] / self.res)) + 1)
                * self.res + self.rangeY[0])

        # Update ranges to make sure that they are correctly represent the meshgrid
        self.rangeX[0] = self.X[0, 0]
        self.rangeX[1] = self.X[-1, -1]
        self.rangeY[0] = self.Y[0, 0]
        self.rangeY[1] = self.Y[-1, -1]

        print("""Meshgrid with X and Y dimensions was created: 
        X size is %s, Y size is %s.\n""" % (np.shape(self.X), np.shape(self.Y)))

        # Now create the weighed grid and the associated grid of weights
        self.sumWeight = np.zeros(np.shape(self.X))
        self.weighGrid = np.zeros(np.shape(self.X))

        self.std_sumWeight = np.zeros((np.shape(self.X)[0], np.shape(self.X)[1], len(z)))
        self.std_weighGrid = np.zeros((np.shape(self.X)[0], np.shape(self.X)[1], len(z)))

        self.stdGrid = np.zeros(np.shape(self.X))

        # Loop through the data - for now using a for loop
        for i in range(len(z)):
            # Get the location of the data in the grid
            x_grid = np.argwhere(x[i] >= self.X[0, :])[-1][0]
            y_grid = np.argwhere(y[i] >= self.Y[:, 0])[-1][0]

            # Set the location of associated kernel in the grid
            k = np.array([[x_grid - self.rpix, x_grid + (self.rpix + 1)],
                          [y_grid - self.rpix, y_grid + (self.rpix + 1)]])

            # Add the contribution to both the weighed grid as well as
            # the grid of summed weights
            self.sumWeight[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1]] = \
                self.sumWeight[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1]] + \
                obs_weight * self.kWeight

            self.weighGrid[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1]] = \
                self.weighGrid[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1]] + \
                obs_weight * self.kWeight * z[i]

            self.std_sumWeight[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1], i] = \
                self.std_sumWeight[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1], i] + \
                self.kWeight

            self.std_weighGrid[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1], i] = \
                self.std_weighGrid[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1], i] + \
                self.kWeight * z[i]

        for i in range(np.shape(self.X)[0]):
            for j in range(np.shape(self.X)[1]):
                self.stdGrid[i, j] = \
                    np.nanstd(self.std_weighGrid[i, j, :] / self.std_sumWeight[i, j, :])

    # Add an array of observations
    def add(self, x=None, y=None, z=None, obs_weight=1):
        if z is None:
            z = []
        if y is None:
            y = []
        if x is None:
            x = []
        if not np.array([x, y, z]).any():
            return print("Nothing has been added, because x or y or z is an empty array")

        # Make sure that arrays cover the full data extent,
        # expanded by the radius of influence in pixel units to
        # ensure that all data can be fully captured in the arrays

        if self.rangeX[0] == 0 and self.rangeX[1] == 0:
            # There is no data yet - define the dimension of the grid so
            # that all data can be held

            self.rangeX[0] = np.amin(x) - self.rpix2coord
            self.rangeX[1] = np.amax(x) + self.rpix2coord
            self.rangeY[0] = np.amin(y) - self.rpix2coord
            self.rangeY[1] = np.amax(y) + self.rpix2coord

            # Create a new meshgrid covering the expanded range
            self.X, self.Y = \
                np.meshgrid(
                    np.arange(0, (np.ceil(np.diff(self.rangeX)[0] / self.res)) + 1)
                    * self.res + self.rangeX[0],
                    np.arange(0, (np.ceil(np.diff(self.rangeY)[0] / self.res)) + 1)
                    * self.res + self.rangeY[0])

            self.rangeX[0] = self.X[0, 0]
            self.rangeX[1] = self.X[-1, -1]
            self.rangeY[0] = self.Y[0, 0]
            self.rangeY[1] = self.Y[-1, -1]

            print("""Meshgrid with X and Y dimensions was created: 
            X size is %d, Y size is %d.\n""" % (np.size(self.X), np.size(self.Y)))

            # Now create the weighed grid and the associated grid of
            # weights. Note that we should test for lack of data by 0s in
            # the weights, not in the weighed depths; zero is a meaningful
            # (and often important) value in grids

            self.weighGrid = np.zeros(np.shape(self.X))
            self.sumWeight = np.zeros(np.shape(self.X))
        else:
            # We should increase our grid by these amount of pixels
            # and by amount of self.rpix
            c_min_x = 0
            c_max_x = 0
            c_min_y = 0
            c_max_y = 0

            xmin = np.amin(x) - self.rpix2coord
            xmax = np.amax(x) + self.rpix2coord
            ymin = np.amin(y) - self.rpix2coord
            ymax = np.amax(y) + self.rpix2coord

            # Determine by now how many pixels
            # the grid should be increased
            # in each direction
            if xmin < self.rangeX[0]:
                c_min_x = np.int64(np.ceil((self.rangeX[0] - xmin) / self.res))
            self.rangeX[0] = self.rangeX[0] - c_min_x * self.res

            if xmax > self.rangeX[1]:
                c_max_x = np.int64(np.ceil((xmax - self.rangeX[1]) / self.res))
            self.rangeX[1] = self.rangeX[1] + c_max_x * self.res

            if ymin < self.rangeY[0]:
                c_min_y = np.int64(np.ceil((self.rangeY[0] - ymin) / self.res))
            self.rangeY[0] = self.rangeY[0] - c_min_y * self.res

            if ymax > self.rangeY[1]:
                c_max_y = np.int64(np.ceil((ymax - self.rangeY[1]) / self.res))
            self.rangeY[1] = self.rangeY[1] + c_max_y * self.res

            if c_min_x or c_min_y or c_max_x or c_max_y:
                n = self.Y
                m = self.X

                # Create a new meshgrid covering the expanded range
                self.X, self.Y = \
                    np.meshgrid(
                        np.arange(0, (np.ceil(np.diff(self.rangeX)[0] / self.res)) + 1)
                        * self.res + self.rangeX[0],
                        np.arange(0, (np.ceil(np.diff(self.rangeY)[0] / self.res)) + 1)
                        * self.res + self.rangeY[0])

                print("""Meshgrid with X and Y dimensions was created: 
                        X size is %d, Y size is %d.""" % (np.size(self.X), np.size(self.Y)))

                # Make sure that the stored ranges are
                # an exact representation of the meshgrid
                self.rangeX[0] = self.X[0, 0]
                self.rangeX[1] = self.X[-1, -1]
                self.rangeY[0] = self.Y[0, 0]
                self.rangeY[1] = self.Y[-1, -1]

                # Get a copy of the existing grid
                wg = np.copy(self.weighGrid)
                sw = np.copy(self.sumWeight)

                std_wg = np.copy(self.std_weighGrid)
                std_sw = np.copy(self.std_sumWeight)
                # std = np.copy(self.stdGrid)

                # Create the new weighed grid and associated sum weights
                self.weighGrid = np.zeros(np.shape(self.X))
                self.sumWeight = np.zeros(np.shape(self.X))

                # Handling the standard deviation grid:

                prev_len = np.shape(self.std_sumWeight)[2]
                new_len = prev_len + len(z)

                self.std_sumWeight = np.zeros((np.shape(self.X)[0], np.shape(self.X)[1], new_len))
                self.std_weighGrid = np.zeros((np.shape(self.X)[0], np.shape(self.X)[1], new_len))

                self.stdGrid = np.zeros(np.shape(self.X))

                self.weighGrid[c_min_y:len(self.weighGrid[:, 0]) - c_max_y,
                               c_min_x:len(self.weighGrid[0, :]) - c_max_x] = wg

                self.sumWeight[c_min_y:len(self.weighGrid[:, 0]) - c_max_y,
                               c_min_x:len(self.weighGrid[0, :]) - c_max_x] = sw

                self.std_weighGrid[c_min_y:len(self.weighGrid[:, 0]) - c_max_y,
                                   c_min_x:len(self.weighGrid[0, :]) - c_max_x, 0:prev_len] = std_wg

                self.std_sumWeight[c_min_y:len(self.weighGrid[:, 0]) - c_max_y,
                                   c_min_x:len(self.weighGrid[0, :]) - c_max_x, 0:prev_len] = std_sw

                # Loop through the data - for now using a for loop
                for i in range(len(z)):
                    # Get the location of the data in the grid

                    x_grid = np.argwhere(x[i] >= self.X[1, :])[-1][0]
                    y_grid = np.argwhere(y[i] >= self.Y[:, 1])[-1][0]

                    # Set the location of associated kernel in the grid
                    k = np.array([[x_grid - self.rpix, x_grid + (self.rpix + 1)],
                                  [y_grid - self.rpix, y_grid + (self.rpix + 1)]])

                    # Add the contribution to both the weighed grid as well as
                    # the grid of summed weights
                    self.sumWeight[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1]] = \
                        self.sumWeight[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1]] + \
                        obs_weight * self.kWeight

                    self.weighGrid[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1]] = \
                        self.weighGrid[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1]] + \
                        obs_weight * self.kWeight * z[i]

                    self.std_sumWeight[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1], prev_len + i] = \
                        self.std_sumWeight[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1], prev_len + i] + \
                        self.kWeight

                    self.std_weighGrid[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1], prev_len + i] = \
                        self.std_weighGrid[k[1, 0]:k[1, 1], k[0, 0]:k[0, 1], prev_len + i] + \
                        self.kWeight * z[i]

                for i in range(np.shape(self.X)[0]):
                    for j in range(np.shape(self.X)[1]):
                        self.stdGrid[i, j] = \
                            np.nanstd(self.std_weighGrid[i, j, :] / self.std_sumWeight[i, j, :])

    # AofI - Area of influence. That method is about beam footpring calculation for every sounding.
    # Depending on the sounding, its position, elevation angle and seafloor surface algorithm should
    # build intersection of the beam cone with a seafloor.
    # The algorithm will be iterative. It will calculate a n number of beam footprint and seafloor
    # assumptions in steps:
    # 1. Assuming that every sounding has vertical incidence and flat bottom
    # 2. Assuming tilted beams and flat bottom
    # 3. Assuming tilted beams and DTM (here I should introduce iterative DTM calculation)
    # 4. Assuming tilted beams, DTM and beam pattern (Beam pattern estimation? Also iterative?)
    # 5. Assuming tilted beams, DTM, beam pattern and water column SV (beam refraction)
    # Steps 3-5 I'm not sure how to implement D:
    #
    # Important to note: beam footprint should play a huge role in sounding weighting
    # because of beam footprint intersections and etc
    #
    # Also, for DTM calculation I should implement convergence criterion, the property of
    # DTM's quality, that will stop iterations at some point.
    # First thoughts about convergence criterion:
    # 1. How much changes are occurring in particular cell. How the depth in that cell is changing,
    # like if it less than some limit, then we done our iterations
    # 2. Those particular cells are grid nodes that affected by footprint (included in beam footprint)
    def area_of_influence(self, x, y, bh, h):
        # x is Easting, 1D array
        # y is Northing, 1D array
        # bh is depression angle in radians
        # h is heading in radians

        aoi = None
        if not np.array([x, y]).any():
            return print("X or Y array is empty! Method is not initialized")

        nS = len(x)

        if len(y) != nS:
            raise RuntimeError("Error! Vectors for Y and X range don't have same size!")

        # Determine whether the current location is covered by the grid

        if not self.rangeX[0]:
            return print("Range X doesn't exists. Method is not initialized")
        else:
            if np.array([np.amin(x) <= self.rangeX[0],
                         np.amax(x) >= self.rangeX[1],
                         np.amin(y) <= self.rangeY[0],
                         np.amax(y) >= self.rangeY[1]]).any():
                return print("X and Y ranges are not correct. They are less than min/max x and y values.\n \
                Method is not initialized")

        # Loop through the data - for now using a for loop
        for i in range(nS):
            # Get the location of the data in the grid
            x_grid = np.argwhere(x[i] >= self.X[1, :])[-1][0]
            y_grid = np.argwhere(y[i] >= self.Y[:, 1])[-1][0]

            # x_grid is now your column, y_grid is your row
            print("x_grid: %d" % x_grid)
            print("y_grid: %d" % y_grid)

            # To demonstrate the indexing look at the value from this location
            d_two = self.weighGrid[y_grid, x_grid] / self.sumWeight[y_grid, x_grid]

            # Now look at one dimensional indexing of the same grid
            ii = np.unravel_index((y_grid + (x_grid - 1) * np.shape(self.X)[0]), np.shape(self.X), 'F')
            d_one = self.weighGrid[ii] / self.sumWeight[ii]

            print("Depth using 2d indexing: %.2f" % d_two)
            print("Depth using 1d indexing: %.2f" % d_one)

            # Original description from Matlab script
            #                 % So ii is the index of the vertex on the lower left of the
            #                 % pixel in which the swath intersects the bottom i.e. the
            #                 % pixels is surrounded by the vertices
            #                 % LL=(y_grid,x_grid)
            #                 % LR=(y_grid,x_grid+1)
            #                 % UL=(y_grid+1,x_grid)
            #                 % UR=(y_grid+1,x_grid+1?)

            #                 % Calculate the direction vector and normalize it, also
            #                 % offset the heading by 90 degrees
            h = h + np.pi / 2
            u = np.sqrt(np.array([np.cos(h) ** 2 * (1 - np.sin(bh[i]) ** 2),
                                  np.sin(h) ** 2 * (1 - np.sin(bh[i]) ** 2),
                                  np.sin(bh[i])]))

        #                 % Note you have to careful about the signs
        #                 % I did not check this for you
        #                 % tan(asin(u(1,3)))*abs(cross distance) should get you
        #                 % approximately to the depth observed

        #                 % etc

    def filter_sd(self, size, crit_angle=1):

        if not np.array([size % 2, size < 1, np.floor(size) != size]).any():
            raise RuntimeError("Kernel size must be an uneven integer greater than 1")
        # Original description from Matlab script
        #             % VERY simple minded approach to get rid of the most significant
        #             % spikes - note that it does not deal with the edges - update
        #             % the algorithm to do that - this is very SLOW, but works
        n_nan = size ** 2 / 2 + 1  # num nan
        size = np.floor(size / 2)

        # Create a sizeXsize normalized filter
        m, n = np.shape(self.X)

        dtm = self.weighGrid / self.sumWeight
        dtm_m = np.zeros((m, n))
        dtm_sd = np.zeros((m, n))

        # By default mask all data
        # self.mask = np.ones((m*n, 1))
        self.mask = np.ones((m, n))

        for i in np.arange(size + 1, m - size + 1, dtype=np.int):
            for j in np.arange(size + 1, n - size + 1, dtype=np.int):
                r = np.arange(i - size, i + size + 1, dtype=np.int)
                c = np.arange(j - size, j + size + 1, dtype=np.int)
                k = dtm[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
                print(k)

                #             % If there are too many nan's don't bother calculating
                #             % This has the risk of masking all the borders so allow
                #             % up to half the numbers+1 to be nan's
                if np.nansum(~np.isnan(k)) > n_nan:
                    index = np.unravel_index(i + (j - 1) * m + 1, np.shape(self.mask), 'F')
                    self.mask[index] = 0  # unmask the data here
                    dtm_m[i, j] = np.nanmean(k)
                    dtm_sd[i, j] = np.nanstd(k)

        # The mean standard deviation
        dtm_sd_nan = np.copy(dtm_sd)
        dtm_sd_nan[dtm_sd_nan == 0] = np.nan
        m_sd = np.nanmean(dtm_sd_nan)

        # Standard Deviation Grid
        self.dtm_sd = dtm_sd_nan

        # Also mask the locations where the standard deviation gets too out
        # of hand i.e. the really big isolated spikes
        self.mask = np.logical_or(self.mask, dtm_sd > 3 * crit_angle * m_sd)

        # Finally, mask the locations where the std is greater than crit times
        # the mean standard deviation
        self.mask = np.logical_or(self.mask, np.abs(dtm_m - dtm) > crit_angle * dtm_sd)

    def filter_diff(self, dtm, crit):
        return  # In progress...

    # Important note: After flat seafloor and non-vert angle incidence scenario implementation in area of
    # influence method, I should create a method for adding and subtracting terrain models with different
    # resolution. The most important stuff is a propagation of the uncertainties associated with DTMs

    def grid_difference(self, dtm1, dtm2):
        return  # In progress...

    # Original description from Matlab script
    # %         function FilterDiff(obj,dtm,crit)
    # %             % Filter by a difference with another terrain model
    # %             % if the difference is too big delete the data
    # %             % This allows you to easily filter crazy canopy heights
    # %
    # %             [m,n]=size(obj.X);
    # %
    # %             if ~all([m,n]==size(dtm.X))
    # %                 error('Grids must match in size!')
    # %             end
    # %
    # %             % Determine the mean difference
    # %
    # %             % Stopped here as the canopy and bottom dtm's are different
    # %             % sizes - you could calculate the bottom dtm and then the
    # %             % canopy dtm - I'm not sure how you do the differencing?
    # %
    # %         end
    def plot(self, mask=None, exag=6, varargin=None):

        # Original description from Matlab script
        #         % This is certainly not the fastest way to plot the data (that
        #         % would be using the surface functions) but it does allow for
        #         % easier manipulation of the data - note that adding lighting
        #         % to this would be nice and slick - this will not be hard to do
        #         dtm = (self.weighGrid / self.sumWeight)*exag

        # Set up some common objects for plotting
        self.dtm = self.weighGrid / self.sumWeight

        x_label = "Easting [m]"
        y_label = "Northing [m]"
        title1 = "Digital Terrain Model"
        title2 = "Standard Deviation Model"

        if mask is not None:
            self.dtm[mask] = np.nan
            title1 = "Masked Digital Terrain Model"

        # 2d plots
        fig, axs = plt.subplots(2, 1)
        fig.set_figheight(5)
        fig.set_figwidth(10)

        # Digital Terrain Model 2D surface
        plot1 = axs[0].pcolormesh(self.X, self.Y, self.dtm,
                                  cmap='gist_rainbow_r', vmin=np.nanmin(self.dtm), vmax=np.nanmax(self.dtm))

        axs[0].ticklabel_format(useOffset=False, style='plain')
        axs[0].set_title(title1)
        axs[0].set_xlabel(x_label)
        axs[0].set_ylabel(y_label)

        # Standard Deviation Model 2D surface
        plot2 = axs[1].pcolormesh(self.X, self.Y, self.stdGrid,
                                  cmap='gist_rainbow_r', vmin=np.nanmin(self.stdGrid), vmax=np.nanmax(self.stdGrid))

        axs[1].ticklabel_format(useOffset=False, style='plain')
        axs[1].set_title(title2)
        axs[1].set_xlabel(x_label)
        axs[1].set_ylabel(y_label)

        # Add colorbars for each plot
        fig.colorbar(plot1, ax=axs[0])
        fig.colorbar(plot2, ax=axs[1])

        # Tight figure's layout
        fig.tight_layout()
        plt.show()
