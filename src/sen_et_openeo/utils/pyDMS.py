# -*- coding: utf-8 -*-
"""
@author: Radoslaw Guzinski
Copyright: (C) 2017, Radoslaw Guzinski
"""

import os
from pathlib import Path
import gc

import numpy as np
from osgeo import gdal
from sklearn import tree, linear_model, ensemble, preprocessing
import sklearn.neural_network as ann_sklearn
from loguru import logger

from sen_et_openeo.utils import pyDMSUtils as utils
from sen_et_openeo.utils.geoloader import (get_band_metadata,
                                           getrasterinfo,
                                           writeraster)


REG_sknn_ann = 0
REG_sklearn_ann = 1


class DecisionTreeRegressorWithLinearLeafRegression(tree.DecisionTreeRegressor):
    ''' Decision tree regressor with added linear (bayesian ridge) regression
    for all the data points falling within each decision tree leaf node.

    Parameters
    ----------
    linearRegressionExtrapolationRatio: float (optional, default: 0.25)
        A limit on extrapolation allowed in the per-leaf linear regressions.
        The ratio is multiplied by the range of values present in each leaves'
        training dataset and added (substracted) to the maxiumum (minimum)
        value.

    decisionTreeRegressorOpt: dictionary (optional, default: {})
        Options to pass to DecisionTreeRegressor constructor. See
        http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
        for possibilities.

    Returns
    -------
    None
    '''

    def __init__(self, linearRegressionExtrapolationRatio=0.25, decisionTreeRegressorOpt={}):
        super(DecisionTreeRegressorWithLinearLeafRegression,
              self).__init__(**decisionTreeRegressorOpt)
        self.decisionTreeRegressorOpt = decisionTreeRegressorOpt
        self.leafParameters = {}
        self.linearRegressionExtrapolationRatio = linearRegressionExtrapolationRatio

    def fit(self, X, y, sample_weight, fitOpt={}):
        ''' Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X: array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            dtype=np.float32 and if a sparse matrix is provided to a sparse
            csc_matrix.

        y: array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (real numbers). Use dtype=np.float64 and
            order='C' for maximum efficiency.

        sample_weight: array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        fitOpt: dictionary (optional, default: {})
            Options to pass to DecisionTreeRegressor fit function. See
            http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
            for possibilities.

        Returns
        -------
        Self
        '''

        # Fit a normal regression tree
        super(DecisionTreeRegressorWithLinearLeafRegression, self).fit(X, y, sample_weight,
                                                                       **fitOpt)

        # Create a linear regression for all input points which fall into
        # one output leaf
        predictedValues = super(
            DecisionTreeRegressorWithLinearLeafRegression, self).predict(X)
        leafValues = np.unique(predictedValues)
        for value in leafValues:
            ind = predictedValues == value
            leafLinearRegrsion = linear_model.BayesianRidge()
            leafLinearRegrsion.fit(X[ind, :], y[ind])
            self.leafParameters[value] = {"linearRegression": leafLinearRegrsion,
                                          "max": np.max(y[ind]),
                                          "min": np.min(y[ind])}

        return self

    def predict(self, X, predictOpt={}):
        ''' Predict class or regression value for X.

        Parameters
        ----------
        X: array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            dtype=np.float32 and if a sparse matrix is provided to a sparse
            csr_matrix.

        predictOpt: dictionary (optional, default: {})
            Options to pass to DecisionTreeRegressor predict function. See
            http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
            for possibilities.

        Returns
        -------
        y: array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        '''

        # Do normal regression tree prediction
        y = super(DecisionTreeRegressorWithLinearLeafRegression,
                  self).predict(X, **predictOpt)

        # And also apply per-leaf linear regression
        for leafValue in self.leafParameters.keys():
            ind = y == leafValue
            if X[ind, :].size > 0:
                y[ind] = self.leafParameters[leafValue]["linearRegression"].predict(
                    X[ind, :])
                # Limit extrapolation
                extrapolationRange = self.linearRegressionExtrapolationRatio * (
                    self.leafParameters[leafValue]["max"] -
                    self.leafParameters[leafValue]["min"])
                y[ind] = np.maximum(y[ind],
                                    self.leafParameters[leafValue]["min"] - extrapolationRange)
                y[ind] = np.minimum(y[ind],
                                    self.leafParameters[leafValue]["max"] + extrapolationRange)

        return y


class DecisionTreeSharpener(object):
    ''' Decision tree based sharpening (disaggregation) of low-resolution
    images using high-resolution images. The implementation is mostly based on [Gao2012].

    Decision tree based regressor is trained with high-resolution data resampled to
    low resolution and low-resolution data and then applied
    directly to high-resolution data to obtain high-resolution representation
    of the low-resolution data.

    The implementation includes selecting training data based on homogeneity
    statistics and using the homogeneity as weight factor ([Gao2012], section 2.2),
    performing linear regression with samples located within each regression
    tree leaf node ([Gao2012], section 2.1), using an ensemble of regression trees
    ([Gao2012], section 2.1), performing local (moving window) and global regression and
    combining them based on residuals ([Gao2012] section 2.3) and performing residual
    analysis and bias correction ([Gao2012], section 2.4)


    Parameters
    ----------
    highResFiles: list of strings
        A list of file paths to high-resolution images to be used during the
        training of the sharpener.

    lowResFiles: list of strings
        A list of file paths to low-resolution images to be used during the
        training of the sharpener. There must be one low-resolution image
        for each high-resolution image.

    lowResQualityFiles: list of strings (optional, default: [])
        A list of file paths to low-resolution quality images to be used to
        mask out low-quality low-resolution pixels during training. If provided
        there must be one quality image for each low-resolution image.

    lowResGoodQualityFlags: list of integers (optional, default: [])
        A list of values indicating which pixel values in the low-resolution
        quality images should be considered as good quality.

    cvHomogeneityThreshold: float (optional, default: 0)
        A threshold of coeficient of variation below which high-resolution
        pixels resampled to low-resolution are considered homogeneous and
        usable during the training of the disaggregator. If threshold is 0 or
        negative then it is set automatically such that 80% of pixels are below
        it.

    movingWindowSize: integer (optional, default: 0)
        The size of local regression moving window in low-resolution pixels. If
        set to 0 then only global regression is performed.

    disaggregatingTemperature: boolean (optional, default: False)
        Flag indicating whether the parameter to be disaggregated is
        temperature (e.g. land surface temperature). If that is the case then
        at some points it needs to be converted into radiance. This is becasue
        sensors measure energy, not temperature, plus radiance is the physical
        measurements it makes sense to average, while radiometric temperature
        behaviour is not linear.

    perLeafLinearRegression: boolean (optional, default: True)
        Flag indicating if linear regression should be performed on all data
        points falling within each regression tree leaf node.

    linearRegressionExtrapolationRatio: float (optional, default: 0.25)
        A limit on extrapolation allowed in the per-leaf linear regressions.
        The ratio is multiplied by the range of values present in each leaves'
        training dataset and added (substracted) to the maxiumum (minimum)
        value.

    regressorOpt: dictionary (optional, default: {})
        Options to pass to DecisionTreeRegressor constructor See
        http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
        for possibilities. Note that max_leaf_nodes and min_samples_leaf
        parameters will beoverwritten in the code.

    baggingRegressorOpt: dictionary (optional, default: {})
        Options to pass to BaggingRegressor constructor. See
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
        for possibilities.

    Returns
    -------
    None


    References
    ----------
    .. [Gao2012] Gao, F., Kustas, W. P., & Anderson, M. C. (2012). A Data
       Mining Approach for Sharpening Thermal Satellite Imagery over Land.
       Remote Sensing, 4(11), 3287–3319. https://doi.org/10.3390/rs4113287
    '''

    def __init__(self,
                 highResFiles,
                 lowResFiles,
                 highresbandnames,
                 lowResQualityFiles=[],
                 lowResGoodQualityFlags=[],
                 cvHomogeneityThreshold=0,
                 movingWindowSize=0,
                 disaggregatingTemperature=False,
                 perLeafLinearRegression=True,
                 linearRegressionExtrapolationRatio=0.25,
                 regressorOpt={},
                 baggingRegressorOpt={}):

        self.highResFiles = highResFiles
        self.lowResFiles = lowResFiles
        self.highresbandnames = highresbandnames
        self.lowResQualityFiles = lowResQualityFiles
        self.lowResGoodQualityFlags = lowResGoodQualityFlags

        if len(self.highResFiles) != len(self.lowResFiles):
            print(
                "There must be a matching high resolution file for each low resolution file")
            raise IOError

        if len(self.lowResQualityFiles) == 0 or \
           (len(self.lowResQualityFiles) == 1 and self.lowResQualityFiles[0] == ""):
            self.useQuality_LR = False
        else:
            self.useQuality_LR = True

        if self.useQuality_LR and len(self.lowResQualityFiles) != len(self.lowResFiles):
            print("The number of quality files must be 0 or the same as number of low " +
                  "resolution files")
            raise IOError

        self.cvHomogeneityThreshold = cvHomogeneityThreshold
        # If threshold is 0 or negative then it is set automatically such that
        # 80% of pixels are below it.
        if self.cvHomogeneityThreshold <= 0:
            self.autoAdjustCvThreshold = True
            self.precentileThreshold = 80
        else:
            self.autoAdjustCvThreshold = False

        # Moving window size in low resolution pixels
        self.movingWindowSize = float(movingWindowSize)
        # The extension (on each side) by which sampling window size is larger
        # then prediction window size (see section 2.3 of Gao paper)
        if self.movingWindowSize % 2 != 0:
            self.movingWindowExtension = self.movingWindowSize * 2
        else:
            self.movingWindowExtension = self.movingWindowSize * 1
        self.windowExtents = []

        self.disaggregatingTemperature = disaggregatingTemperature

        # Flag to determine whether a multivariate linear regression should be
        # constructed for samples in each leaf of the regression tree
        # (see section 2.1 of Gao paper)
        self.perLeafLinearRegression = perLeafLinearRegression
        self.linearRegressionExtrapolationRatio = linearRegressionExtrapolationRatio

        self.regressorOpt = regressorOpt
        self.baggingRegressorOpt = baggingRegressorOpt

    def trainSharpener(self):
        ''' Train the sharpener using high- and low-resolution input files
        and settings specified in the constructor. Local (moving window) and
        global regression decision trees are trained with high-resolution data
        resampled to low resolution and low-resolution data. The training
        dataset is selected based on homogeneity of resampled high-resolution
        data being below specified threshold and quality mask (if given) of
        low resolution data. The homogeneity statistics are also used as weight
        factors for the training samples (more homogenous - higher weight).

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Select good data (training samples) from low- and high-resolution
        # input images.

        nfiles = len(self.highResFiles)
        i = 0

        # prepare outputs
        goodData_LR = [None for _ in range(nfiles)]
        goodData_HR = [None for _ in range(nfiles)]
        weight = [None for _ in range(nfiles)]

        # extract data from files
        for highResFile, lowResFile in zip(self.highResFiles,
                                           self.lowResFiles):

            scene_HR = gdal.Open(highResFile)
            scene_LR = gdal.Open(lowResFile)

            # First subset and reproject low res scene to fit with
            # high res scene
            subsetScene_LR = utils.reprojectSubsetLowResScene(
                scene_HR, scene_LR)
            gt_LR = subsetScene_LR.GetGeoTransform()
            data_LR = subsetScene_LR.GetRasterBand(1).ReadAsArray()
            scale, dtype, nodata = get_band_metadata('S3-LST')
            # scale and apply nodata value
            data_LR = data_LR.astype(np.float32)
            data_LR[data_LR == nodata] = np.nan
            data_LR = data_LR / scale

            # Do the same with low res quality file (if provided) and flag
            # pixels which are considered to be of good quality
            if self.useQuality_LR:
                quality_LR = gdal.Open(self.lowResQualityFiles[i])
                subsetQuality_LR = utils.reprojectSubsetLowResScene(
                    scene_HR, quality_LR)
                subsetQualityMask = subsetQuality_LR.GetRasterBand(
                    1).ReadAsArray()
                qualityPix = np.in1d(subsetQualityMask.ravel(),
                                     self.lowResGoodQualityFlags).reshape(
                                         subsetQualityMask.shape)
                quality_LR = None
            else:
                qualityPix = np.ones(data_LR.shape).astype(bool)

            # Low resolution pixels with NaN value are always of bad quality
            qualityPix = np.logical_and(qualityPix, ~np.isnan(data_LR))

            # Then resample high res scene to low res pixel size while
            # extracting sub-low-res-pixel homogeneity statistics
            resMean, resStd = utils.resampleHighResToLowRes(
                scene_HR, subsetScene_LR, self.highresbandnames)
            resMean[resMean == 0] = 0.000001
            resCV = np.sum(resStd/resMean, 2) / resMean.shape[2]
            resCV[np.isnan(resCV)] = 1000
            epsg, bounds = getrasterinfo(lowResFile)[0:2]
            cvfile = lowResFile.replace('S3-LST', 'S3-CV')
            Path(cvfile).parent.mkdir(exist_ok=True, parents=True)
            writeraster(resCV, cvfile, epsg, bounds)

            # Resampled high resolution pixels where at least one "parameter"
            # is NaN are also of bad quality
            resNaN = np.any(np.isnan(resMean), -1)
            qualityPix = np.logical_and(qualityPix, ~resNaN)

            # Now extract the good quality low res and
            # high res pixels
            window = [0, data_LR.shape[0], 0, data_LR.shape[1]]
            rows = slice(window[0], window[1])
            cols = slice(window[2], window[3])
            qualityPixWindow = qualityPix[rows, cols]
            resCVWindow = resCV[rows, cols]

            # Good pixels are those where low res data quality
            # is good and high res data is homonogenous
            if self.autoAdjustCvThreshold:
                g = np.logical_and.reduce((qualityPixWindow,
                                           resCVWindow < 1000,
                                           resCVWindow > 0))
                if ~np.any(g):
                    self.cvHomogeneityThreshold = 0
                else:
                    self.cvHomogeneityThreshold = np.percentile(
                        resCVWindow[g], self.precentileThreshold)
                logger.info('Homogeneity CV threshold: %.2f' %
                            self.cvHomogeneityThreshold)

            homogenousPix = np.logical_and(
                resCVWindow < self.cvHomogeneityThreshold, resCVWindow > 0)
            goodPix = np.logical_and(homogenousPix, qualityPixWindow)

            goodData_LR[i] = utils.appendNpArray(goodData_LR[i],
                                                 data_LR[rows, cols]
                                                 [goodPix])
            goodData_HR[i] = utils.appendNpArray(goodData_HR[i],
                                                 resMean[rows, cols,
                                                         :][goodPix, :],
                                                 axis=0)

            # Also estimate weight given to each pixel as the
            # inverse of its heterogeneity
            w = 1/resCVWindow[goodPix]
            weight[i] = utils.appendNpArray(weight[i], w)

            # Print some stats
            if np.prod(data_LR[rows, cols][qualityPixWindow].shape) > 0:
                percentageUsedPixels = int(float(np.prod(
                    goodData_LR[i].shape)) / float(np.prod(
                        data_LR[rows, cols][qualityPixWindow].shape)
                ) * 100)
                nUsedPixels = np.prod(goodData_LR[i].shape)
                logger.info(f'Number of training pixels is {nUsedPixels}'
                            f' representing {percentageUsedPixels}'
                            ' percent of available low-resolution data.')

            # Close all files
            scene_HR = None
            scene_LR = None
            subsetScene_LR = None
            if self.useQuality_LR:
                subsetQuality_LR = None
            i += 1

        # Merge all datasets to one big dataset
        goodData_LR = np.concatenate(goodData_LR)
        goodData_HR = np.concatenate(goodData_HR)
        weight = np.concatenate(weight)

        # Once all the samples have been picked fit the regression
        local = False
        self.reg = [None]
        self.reg[0] = self._doFit(goodData_LR, goodData_HR,
                                  weight, local)

    def applySharpener(self, highResFilename, lowResFilename=None):
        ''' Apply the trained sharpener to a given high-resolution image to
        derive corresponding disaggregated low-resolution image. If local
        regressions were used during training then they will only be applied
        where their moving window extent overlaps with the high resolution
        image passed to this function. Global regression will be applied to the
        whole high-resolution image wihtout geographic constraints.

        Parameters
        ----------
        highResFilename: string
            Path to the high-resolution image file do be used during
            disaggregation.

        lowResFilename: string (optional, default: None)
            Path to the low-resolution image file corresponding to the
            high-resolution input file. If local regressions
            were trained and low-resolution filename is given then the local
            and global regressions will be combined based on residual values of
            the different regressions to the low-resolution image
            (see [Gao2012] 2.3).
            If local regressions were trained and low-resolution
            filename is not given then only the local regressions will be used.


        Returns
        -------
        outImage: GDAL memory file object
            The file object contains an in-memory, georeferenced disaggregator
            output.
        '''

        # Open and read the high resolution input file
        highResFile = gdal.Open(highResFilename)
        inData = np.zeros((highResFile.RasterYSize, highResFile.RasterXSize,
                           highResFile.RasterCount))
        for band in range(highResFile.RasterCount):
            data = highResFile.GetRasterBand(
                band+1).ReadAsArray()
            # get band name and associated metadata
            scale, dtype, nodata = get_band_metadata(
                self.highresbandnames[band])
            # apply scaling and nodata value
            data = data.astype(np.float32)
            data[data == nodata] = np.nan
            if scale is not None:
                data = data / scale
            inData[:, :, band] = data

        # Temporarly get rid of NaN's
        nanInd = np.isnan(inData)
        inData[nanInd] = 0
        inData = inData.astype(np.float32)
        # Do the downscailing on the whole input image
        outData = self._doPredict(inData, self.reg[0])

        # Fix NaN's
        nanInd = np.any(nanInd, -1)
        outData[nanInd] = np.nan

        # # save to memory
        # outImage = utils.saveImg(outData,
        #                          highResFile.GetGeoTransform(),
        #                          highResFile.GetProjection(),
        #                          "MEM",
        #                          noDataValue=np.nan)

        highResFile = None
        inData = None
        gc.collect()

        return outData

    def residualAnalysis(self, disaggregatedFile, lowResFilename,
                         lowResQualityFilename=None, doCorrection=True):
        ''' Perform residual analysis and (optional) correction on the
        disaggregated file (see [Gao2012] 2.4).

        Parameters
        ----------
        disaggregatedFile: string or GDAL file object
            If string, path to the disaggregated image file; if gdal file
            object, the disaggregated image.

        lowResFilename: string
            Path to the low-resolution image file corresponding to the
            high-resolution disaggregated image.

        lowResQualityFilename: string (optional, default: None)
            Path to low-resolution quality image file. If provided then low
            quality values are masked out during residual analysis. Otherwise
            all values are considered to be of good quality.

        doCorrection: boolean (optional, default: True)
            Flag indication whether residual (bias) correction should be
            performed or not.


        Returns
        -------
        residualImage: GDAL memory file object
            The file object contains an in-memory,
            georeferenced residual image.

        correctedImage: GDAL memory file object
            The file object contains an in-memory, georeferenced residual
            corrected disaggregated image, or None if doCorrection was set to
            False.
        '''

        if not os.path.isfile(str(disaggregatedFile)):
            scene_HR = disaggregatedFile
        else:
            scene_HR = gdal.Open(disaggregatedFile)
        scene_LR = gdal.Open(lowResFilename)
        if lowResQualityFilename is not None:
            quality_LR = gdal.Open(lowResQualityFilename)
        else:
            quality_LR = None

        residual_HR, residual_LR, gt_res = self._calculateResidual(
            scene_HR, scene_LR, quality_LR)

        if self.disaggregatingTemperature:
            if doCorrection:
                corrected = (residual_HR +
                             scene_HR.GetRasterBand(1).ReadAsArray()**4)**0.25
                correctedImage = utils.saveImg(corrected,
                                               scene_HR.GetGeoTransform(),
                                               scene_HR.GetProjection(),
                                               "MEM",
                                               noDataValue=np.nan)
            else:
                correctedImage = None
            # Convert residual back to temperature for easier visualisation
            residual_LR = (residual_LR + 273.15**4)**0.25 - 273.15
        else:
            if doCorrection:
                corrected = residual_HR + \
                    (scene_HR.GetRasterBand(1).ReadAsArray())
                correctedImage = utils.saveImg(corrected,
                                               scene_HR.GetGeoTransform(),
                                               scene_HR.GetProjection(),
                                               "MEM",
                                               noDataValue=np.nan)
            else:
                correctedImage = None

        residualImage = utils.saveImg(residual_LR,
                                      gt_res,
                                      scene_HR.GetProjection(),
                                      "MEM",
                                      noDataValue=np.nan)

        print("LR residual bias: "+str(np.nanmean(residual_LR)))
        print("LR residual RMSD: "+str(np.nanmean(residual_LR**2)**0.5))

        scene_HR = None
        scene_LR = None
        quality_LR = None

        return residualImage, correctedImage

    def _doFit(self, goodData_LR, goodData_HR, weight, local):
        ''' Private function. Fits the regression tree.
        '''

        # For local regression constrain the number of tree
        # nodes (rules) - section 2.3
        if local:
            self.regressorOpt["max_leaf_nodes"] = 10
        else:
            self.regressorOpt["max_leaf_nodes"] = 30
        self.regressorOpt["min_samples_leaf"] = 10

        # If per leaf linear regression is used then use modified
        # DecisionTreeRegressor. Otherwise use the standard one.
        if self.perLeafLinearRegression:
            baseRegressor = \
                DecisionTreeRegressorWithLinearLeafRegression(
                    self.linearRegressionExtrapolationRatio,
                    self.regressorOpt)
        else:
            baseRegressor = \
                tree.DecisionTreeRegressor(**self.regressorOpt)

        reg = ensemble.BaggingRegressor(
            baseRegressor, **self.baggingRegressorOpt)
        if goodData_HR.shape[0] <= 1:
            reg.max_samples = 1.0
        reg = reg.fit(goodData_HR, goodData_LR, sample_weight=weight)

        return reg

    def _doPredict(self, inData, reg):
        ''' Private function. Calls the regression tree.
        '''

        origShape = inData.shape
        if len(origShape) == 3:
            bands = origShape[2]
        else:
            bands = 1
        # Do the actual decision tree regression
        inData = inData.reshape((-1, bands))
        outData = reg.predict(inData)
        outData = outData.reshape((origShape[0], origShape[1]))

        return outData

    def _calculateResidual(self, downscaledScene, originalScene,
                           originalSceneQuality=None):
        ''' Private function. Calculates residual between overlapping
            high-resolution and low-resolution images.
        '''

        # First subset and reproject original (low res) scene to fit with
        # downscaled (high res) scene
        subsetScene_LR = utils.reprojectSubsetLowResScene(downscaledScene,
                                                          originalScene)
        data_LR = subsetScene_LR.GetRasterBand(
            1).ReadAsArray()
        scale, dtype, nodata = get_band_metadata('S3-LST')
        # scale and apply nodata value
        data_LR = data_LR.astype(np.float32)
        data_LR[data_LR == nodata] = np.nan
        data_LR = data_LR / scale
        gt_LR = subsetScene_LR.GetGeoTransform()

        # If quality file for the low res scene is provided then mask out all
        # bad quality pixels in the subsetted LR scene.
        # Otherwise assume that all low res pixels are of good quality.
        if originalSceneQuality is not None:
            subsetQuality_LR = utils.reprojectSubsetLowResScene(
                downscaledScene, originalSceneQuality)
            goodPixMask_LR = subsetQuality_LR.GetRasterBand(1).ReadAsArray()
            goodPixMask_LR = np.in1d(goodPixMask_LR.ravel(),
                                     self.lowResGoodQualityFlags).reshape(
                                         goodPixMask_LR.shape)
            data_LR[~goodPixMask_LR] = np.nan

        # Then resample high res scene to low res pixel size
        if self.disaggregatingTemperature:
            # When working with tempratures they should be converted to
            # radiance values before aggregating to be physically accurate.
            radianceScene = utils.saveImg(
                downscaledScene.GetRasterBand(1).ReadAsArray()**4,
                downscaledScene.GetGeoTransform(),
                downscaledScene.GetProjection(),
                "MEM",
                noDataValue=np.nan)
            resMean, _ = utils.resampleHighResToLowRes(radianceScene,
                                                       subsetScene_LR)
            # Find the residual (difference) between the two)
            residual_LR = data_LR**4 - resMean[:, :, 0]
        else:
            resMean, _ = utils.resampleHighResToLowRes(downscaledScene,
                                                       subsetScene_LR)
            # Find the residual (difference) between the two
            residual_LR = data_LR - resMean[:, :, 0]

        # Smooth the residual and resample to high resolution
        residual = utils.binomialSmoother(residual_LR)
        residualDs = utils.saveImg(residual, subsetScene_LR.GetGeoTransform(),
                                   subsetScene_LR.GetProjection(), "MEM",
                                   noDataValue=np.nan)
        residualScene_BL = utils.resampleWithGdalWarp(residualDs,
                                                      downscaledScene,
                                                      resampleAlg="bilinear")
        residualDs = None

        residual = residualScene_BL.GetRasterBand(1).ReadAsArray()
        # Sometimes there can be 1 HR pixel NaN border arond LR invalid
        # pixels due to resampling.
        # Fuction below fixes this.
        # Image border pixels are excluded due to numba stencil limitations.
        residual[1:-1, 1:-1] = utils.removeEdgeNaNs(residual)[1:-1, 1:-1]
        residualScene_BL = None

        # The residual array might be slightly smaller then the
        # downscaled because of the subsetting of the low resolution scene.
        # In that case just pad the missing values with neighbours.
        downscaled = downscaledScene.GetRasterBand(1).ReadAsArray()
        if downscaled.shape != residual.shape:
            temp = np.zeros(downscaled.shape)
            temp[:residual.shape[0], :residual.shape[1]] = residual
            temp[residual.shape[0]:, :] = \
                temp[2*(residual.shape[0] - downscaled.shape[0])
                        :residual.shape[0] - downscaled.shape[0], :]
            temp[:, residual.shape[1]:] = \
                temp[:, 2*(residual.shape[1] - downscaled.shape[1])
                           :residual.shape[1] - downscaled.shape[1]]

            residual = temp

        residualScene = None
        subsetScene_LR = None
        subsetQuality_LR = None

        return residual, residual_LR, gt_LR


class NeuralNetworkSharpener(DecisionTreeSharpener):
    ''' Neural Network based sharpening (disaggregation) of low-resolution
    images using high-resolution images. The implementation is mostly based on [Gao2012] as
    implemented in DescisionTreeSharpener except that Decision Tree regressor is replaced by
    Neural Network regressor.

    Nerual network based regressor is trained with high-resolution data resampled to
    low resolution and low-resolution data and then applied
    directly to high-resolution data to obtain high-resolution representation
    of the low-resolution data.

    The implementation includes selecting training data based on homogeneity
    statistics and using the homogeneity as weight factor ([Gao2012], section 2.2),
    performing linear regression with samples located within each regression
    tree leaf node ([Gao2012], section 2.1), using an ensemble of regression trees
    ([Gao2012], section 2.1), performing local (moving window) and global regression and
    combining them based on residuals ([Gao2012] section 2.3) and performing residual
    analysis and bias correction ([Gao2012], section 2.4)

    Parameters
    ----------
    highResFiles: list of strings
        A list of file paths to high-resolution images to be used during the
        training of the sharpener.

    lowResFiles: list of strings
        A list of file paths to low-resolution images to be used during the
        training of the sharpener. There must be one low-resolution image
        for each high-resolution image.

    lowResQualityFiles: list of strings (optional, default: [])
        A list of file paths to low-resolution quality images to be used to
        mask out low-quality low-resolution pixels during training. If provided
        there must be one quality image for each low-resolution image.

    lowResGoodQualityFlags: list of integers (optional, default: [])
        A list of values indicating which pixel values in the low-resolution
        quality images should be considered as good quality.

    cvHomogeneityThreshold: float (optional, default: 0.25)
        A threshold of coeficient of variation below which high-resolution
        pixels resampled to low-resolution are considered homogeneous and
        usable during the training of the disaggregator.

    movingWindowSize: integer (optional, default: 0)
        The size of local regression moving window in low-resolution pixels. If
        set to 0 then only global regression is performed.

    disaggregatingTemperature: boolean (optional, default: False)
        Flag indicating whether the parameter to be disaggregated is
        temperature (e.g. land surface temperature). If that is the case then
        at some points it needs to be converted into radiance. This is becasue
        sensors measure energy, not temperature, plus radiance is the physical
        measurements it makes sense to average, while radiometric temperature
        behaviour is not linear.

    regressionType: int (optional, default: 0)
        Flag indicating whether scikit-neuralnetwork (flag value = REG_sknn_ann = 0)
        or scikit-learn (flag value = REG_sklearn_ann = 1) implementations of
        nearual network should be used. See
        https://github.com/aigamedev/scikit-neuralnetwork and
        http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
        for details.

    regressorOpt: dictionary (optional, default: {})
        Options to pass to neural network regressor constructor See links in
        regressionType parameter description for details.

    baggingRegressorOpt: dictionary (optional, default: {})
        Options to pass to BaggingRegressor constructor. See
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
        for possibilities.


    Returns
    -------
    None


    References
    ----------
    .. [Gao2012] Gao, F., Kustas, W. P., & Anderson, M. C. (2012). A Data
       Mining Approach for Sharpening Thermal Satellite Imagery over Land.
       Remote Sensing, 4(11), 3287–3319. https://doi.org/10.3390/rs4113287
    '''

    def __init__(self,
                 highResFiles,
                 lowResFiles,
                 lowResQualityFiles=[],
                 lowResGoodQualityFlags=[],
                 cvHomogeneityThreshold=0.25,
                 movingWindowSize=0,
                 disaggregatingTemperature=False,
                 regressionType=REG_sknn_ann,
                 regressorOpt={},
                 baggingRegressorOpt={}):

        super(NeuralNetworkSharpener, self).__init__(highResFiles,
                                                     lowResFiles,
                                                     lowResQualityFiles,
                                                     lowResGoodQualityFlags,
                                                     cvHomogeneityThreshold,
                                                     movingWindowSize,
                                                     disaggregatingTemperature,
                                                     regressorOpt=regressorOpt,
                                                     baggingRegressorOpt=baggingRegressorOpt)
        self.regressionType = regressionType
        # Move the import of sknn here because this library is not easy to
        # install but this shouldn't prevent the use of other parts of pyDMS.
        if self.regressionType == REG_sknn_ann:
            import sknn.mlp as ann_sknn

    def _doFit(self, goodData_LR, goodData_HR, weight, local):
        ''' Private function. Fits the neural network.
        '''

        # Once all the samples have been picked build the regression using
        # neural network approach
        print('Fitting neural network')
        HR_scaler = preprocessing.StandardScaler()
        data_HR = HR_scaler.fit_transform(goodData_HR)
        LR_scaler = preprocessing.StandardScaler()
        data_LR = LR_scaler.fit_transform(goodData_LR.reshape(-1, 1))
        if self.regressionType == REG_sknn_ann:
            layers = []
            if 'hidden_layer_sizes' in self.regressorOpt.keys():
                for layer in self.regressorOpt['hidden_layer_sizes']:
                    layers.append(ann_sknn.Layer(
                        self.regressorOpt['activation'], units=layer))
            else:
                layers.append(ann_sknn.Layer(
                    self.regressorOpt['activation'], units=100))
            self.regressorOpt.pop('activation')
            self.regressorOpt.pop('hidden_layer_sizes')
            output_layer = ann_sknn.Layer('Linear', units=1)
            layers.append(output_layer)
            baseRegressor = ann_sknn.Regressor(layers, **self.regressorOpt)
        else:
            baseRegressor = ann_sklearn.MLPRegressor(**self.regressorOpt)

        # NN regressors do not support sample weights.
        weight = None

        reg = ensemble.BaggingRegressor(
            baseRegressor, **self.baggingRegressorOpt)
        if data_HR.shape[0] <= 1:
            reg.max_samples = 1.0
        reg = reg.fit(data_HR, np.ravel(data_LR), sample_weight=weight)

        return {"reg": reg, "HR_scaler": HR_scaler, "LR_scaler": LR_scaler}

    def _doPredict(self, inData, nn):
        ''' Private function. Calls the neural network.
        '''

        reg = nn["reg"]
        HR_scaler = nn["HR_scaler"]
        LR_scaler = nn["LR_scaler"]

        origShape = inData.shape
        if len(origShape) == 3:
            bands = origShape[2]
        else:
            bands = 1

        # Do the actual neural network regression
        inData = inData.reshape((-1, bands))
        inData = HR_scaler.transform(inData)
        outData = reg.predict(inData)
        outData = LR_scaler.inverse_transform(outData)
        outData = outData.reshape((origShape[0], origShape[1]))

        return outData
