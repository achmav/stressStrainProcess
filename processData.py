#Standard Libraries we Use
import os
import math
import statistics
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
import urllib


class dataStruct():
    def __init__(self,processAll = True, web = False):
        self.web = web
        print(self.web)
        #Blank Dict to hold Samples
        self.samples = []
        #Loading our Compositions
        self._loadCompositions()
        #Creating All Samples
        if processAll:
            self._loadAll()
            self._organizeCompositions()
        self._loadProposals()
        self._pickleSamples()
    #Loads in our Composition File
    def _loadCompositions(self,fileName = 'Compositions.csv'):
        if self.web:
            import urllib
            data = urllib.request.urlopen('https://raw.githubusercontent.com/achmav/stressStrainProcess/master/Compositions.csv').read()
        else:
            #Seeing if our Compositions file is there
            if not fileName in os.listdir('.'):
                raise Exception("Missing Compositions File in Directory " + os.getcwd())
            #Opening the Compositions File
            with open(fileName, 'r') as inFile:
                data = inFile.read()
        #Splitting up the CSV
        self.lines = [i.split(',') for i in data.split('\n') if i]
        self.category = self.lines.pop(0)
        self.header = self.lines.pop(0)
    def _loadAll(self):
        self._unpickleSamples()
        #Making a Dictionary for each Line and turning it into a Sample
        for line in self.lines:
            #Making sure we don't have this datapoint already loaded
            if int(line[0]) not in [i.name for i in self.samples if i.formulation == int(line[1])]:
                self.samples.append(sample(line,self.category,self.header))
    #Organizes Compositions
    def _organizeCompositions(self):
        #Finding the Formulations
        formulationNames = list(set([i.formulation for i in self.samples]))
        self.formulations = [formulation([x for x in self.samples if x.formulation == i]) for i in formulationNames]
    #Makes a Pandas DF of Formulations
    def make_dataframe(self):
        import pandas
        sample_type = []
        for x in [i for i in self.formulations if hasattr(i,'maxStress') and getattr(i,'maxStress')]:
            if x.name in [i for i in range(23,52)]+[i for i in range(57,62)]+[75,76]:
                sample_type.append('Initial Samples')
            elif x.name in [43,44,46,75,76]:
                sample_type.append('Base Components')
            else:
                sample_type.append('TSEMO Composition')

        data = {'maxStress':[getattr(i,'maxStress') for i in self.formulations if hasattr(i,'maxStress') and getattr(i,'maxStress')],
                        'youngSlope':[getattr(i,'youngSlope') for i in self.formulations if hasattr(i,'maxStress') and getattr(i,'youngSlope')],
                        'toughness':[getattr(i,'toughness') for i in self.formulations if hasattr(i,'maxStress') and getattr(i,'toughness')],
                        'sampleType':sample_type}
        return pandas.DataFrame(data=data)
    #Pickling Functions
    def _pickleSamples(self):
        if not self.web:
            with open(os.path.join('DataNew','Samples.p'), 'wb') as outFile:
                #Importing Pickle
                try:
                    import cPickle as pickle
                except:
                    import pickle
                #Dumping our Samples
                pickle.dump(self.samples, outFile)
                print('Saved ' +str(len(self.samples))+ ' Samples to Samples.p')
    #Loading Formulations
    def _unpickleSamples(self):
        #Importing Pickle
        try:
            import cPickle as pickle
        except:
            import pickle
        #If we've got a pickle file, let's load it
        if self.web:
            #Link to our Input Data
            inputUrl = 'https://github.com/achmav/stressStrainProcess/raw/master/DataNew/Samples.p'
            #Loading it in from Putty
            import urllib
            filedata = urllib.request.urlopen(inputUrl)
            self.samples = pickle.load(filedata)
        else:
            if 'Samples.p' in os.listdir('DataNew'):
                with open(os.path.join('DataNew','Samples.p'), 'rb') as inFile:
                    self.samples = pickle.load(inFile)
                print('Loaded in ' + str(len(self.samples)) + ' samples from Cache')
    #Loads in proposed samples
    def _loadProposals(self):
        import os
        #Finding all the proposedSamples files
        subFolders = [x for x in os.walk('.') if any(['proposedSamples' in i for i in x[-1]])]
        subFolders = [[os.path.join(x[0],y) for y in x[-1] if 'proposedSamples' in y] for x in subFolders]
        subFolders = [item for sublist in subFolders for item in sublist]
        #Loading all of our Proposals
        self.proposals = [item for sublist in [proposedSamples(x).lines for x in subFolders] for item in sublist]
        print('Loaded ' + str(len(self.proposals)) + ' proposed Samples in')
        for n in self.formulations:
            n._matchProposal(self.proposals)
    #Exports Formulations
    def saveFormulations(self):
            #Ignore Files
        ignore = [23,24,36,40,54,55,56,57,58,59,60,61,149,150,151,183]
        formulations = [i for i in self.formulations if int(i.name) not in ignore and i.samples and i.maxStrainCV]
        #Dealing with the Formulations that need averaging
        avgForms = [i for i in self.formulations if int(i.name) in [55,56,57,58,59,60,61]]
        #Writing a Summary File with All Data
        with open(os.path.join('Plots','Summary.csv'), 'w') as outFile:
            import csv
            spamwriter = csv.writer(outFile, lineterminator="\n")
            spamwriter.writerow(['Formulation','maxStrain','cv','maxStress','cv','modulus','cv','toughness','cv']
                                    +list(self.formulations[0].processParameters.keys())
                                    +list(self.formulations[0].composition.keys())
                                    + ['proposed stress','proposed modulus','proposed toughness','hypervolume improvement']
                                    + ['stress Err', 'modulus Err', 'toughness Err'])
            for n in formulations:
                spamwriter.writerow([n.name,n.maxStrain,n.maxStrainCV,n.maxStress,n.maxStressCV,n.youngSlope,n.youngSlopeCV,n.toughness,n.toughnessCV]
                                    +[n.processParameters[i] for i in n.processParameters.keys()]
                                    +[n.composition[i] for i in n.composition.keys()]
                                    +[n.proposedStress,n.proposedYoung,n.proposedToughness,n.hypervolumeImprovement]
                                    +[n.stressErr, n.youngErr, n.toughnessErr])
        #Writing the Files for TS_Emo


        #inputSamples File (Compositions)
        with open(os.path.join('Plots','inputSamples.txt'), 'w') as outFile:
            spamwriter = csv.writer(outFile, delimiter='\t', lineterminator='\n')
            spamwriter.writerow([avgForms[0].composition[i] for i in avgForms[0].composition.keys()])
            for n in formulations:
                spamwriter.writerow([n.composition[i] for i in n.composition.keys()])
        #outputData File (Results)
        with open(os.path.join('Plots','outputData.txt'), 'w') as outFile:
            spamwriter = csv.writer(outFile, delimiter='\t', lineterminator='\n')
            spamwriter.writerow([sum([i.maxStress for i in avgForms])/len(avgForms),sum([i.youngSlope for i in avgForms])/len(avgForms),sum([i.toughness for i in avgForms])/len(avgForms)])
            for n in formulations:
                spamwriter.writerow([n.maxStress,n.youngSlope,n.toughness])

    #Loads ONE Sample
    def loadOne(self,s,f):
        dataLine = [i for i in self.lines if int(i[self.header.index('Sample')]) == int(s)
                                and int(i[self.header.index('Formulation')]) == int(f)]
        if dataLine:
            self.samples.append(sample(dataLine[0],self.category,self.header))
        else:
            print('No Sample Found')

#Saves a Formulation
class formulation():
    def __init__(self,samples):
        self.samples = samples
        self.color = 'r'
    """Providing our Base Properties that we care about as Properties"""
    @property
    def name(self):
        return self.samples[0].formulation
    @property
    def processParameters(self):
        return self.samples[0].processParameters
    @property
    def composition(self):
        return self.samples[0].composition
    #{'maxStrain':'Elongation At Break','maxStress':'Tensile Strength','youngSlope':'Youngs Modulus','toughness':'Toughness'}
    @property
    def maxStrain(self):
        return self._returnMean('maxStrain')
    @property
    def maxStress(self):
        return self._returnMean('maxStress')
    @property
    def youngSlope(self):
        return self._returnMean('youngSlope')
    @property
    def toughness(self):
        return self._returnMean('toughness')
    #CVs
    @property
    def maxStrainCV(self):
        return self._returnCV('maxStrain')
    @property
    def maxStressCV(self):
        return self._returnCV('maxStress')
    @property
    def youngSlopeCV(self):
        return self._returnCV('youngSlope')
    @property
    def toughnessCV(self):
        return self._returnCV('toughness')
    """Function to Match up with Proposed Data"""
    def _matchProposal(self,listOfProposals):
        compositionList = [self.composition[i] for i in self.composition.keys()]
        matchedProposal = [i for i in listOfProposals if i.composition == compositionList]
        if matchedProposal:
            self.proposedStress = matchedProposal[0].performance[0]
            self.proposedYoung = matchedProposal[0].performance[1]
            self.proposedToughness = matchedProposal[0].performance[2]
            self.hypervolumeImprovement = matchedProposal[0].hypervolume
        else:
            self.proposedStress = 0
            self.proposedYoung = 0
            self.proposedToughness = 0
            self.hypervolumeImprovement = 0
        self._calculateErrors()
    #Calculates the Error between Proposed Results and Actual Results
    def _calculateErrors(self):
        if self.proposedStress and self.maxStress:
            self.stressErr = abs(self.proposedStress - self.maxStress)/(abs(self.proposedStress + self.maxStress)/2)
            self.youngErr = abs(self.proposedYoung - self.youngSlope)/(abs(self.proposedYoung + self.youngSlope)/2)
            self.toughnessErr = abs(self.proposedToughness - self.toughness)/(abs(self.proposedToughness + self.toughness)/2)
        else:
            self.stressErr = 0
            self.youngErr = 0
            self.toughnessErr = 0
    """Helpful Functions"""
    def _returnMean(self,attribute):
        if any([hasattr(i,attribute) for i in self.samples]):
            return numpy.mean([getattr(i,attribute) for i in self.samples if hasattr(i,attribute)])
        else:
            return None
    def _returnCV(self,attribute):
        if any([hasattr(i,attribute) for i in self.samples]):
            return statistics.stdev([getattr(i,attribute) for i in self.samples if hasattr(i,attribute)])/getattr(self,attribute)
        else:
            return None


#Holds information about our Samples we've generated
#Takes in a value dict from the File Parser
class sample():
    def __init__(self,line,category,header):
        #Finding our Composition information
        self.composition = self._valueDict(line,header,category,'Composition')
        #Process Parameters
        self.processParameters = self._valueDict(line,header,category,'Process')
        #Finding the other attributes
        self.sampleParams = self._valueDict(line,header,category,'Sample')
        self.formulation = self.sampleParams['Formulation']
        self.name = self.sampleParams['Sample']
        #Our Functions to Process Data
        self._sampleGeometry()
        self._loadStressStrain()
        #If we have a Stress Data File
        if self.stressData:
            self._physicalCharacteristics()
            self._cropNew()
    """File Operations"""
    #Parses Stress Strain Data
    def _loadStressStrain(self):
        #If we've got a Formulation, see if we can use the Formulation Sample Name
        if self.formulation:
            filePath = os.path.join('DataNew','F'+str(self.formulation)+'S'+str(self.name)+'.csv')
        else:
            #Looking to see if we've got the file
            filePath = os.path.join('Data','Specimen_RawData_'+str(self.name)+'.csv')
        if os.path.exists(filePath):
            if self.formulation:
                print('Loaded Stress Strain Data for F'+str(self.formulation) + 'S' + str(self.name))
            else:
                print('Loaded Stress Strain Data for ' + str(self.name))
            loadFile = stressDataFile(filePath)
            self.stressData = loadFile.data
        else:
            if self.formulation:
                print('No Stress Strain Data for F'+str(self.formulation) + 'S' + str(self.name))
            else:
                print('No Stress data found for Sample ' + str(self.name))
            self.stressData = None
        #Place Holder for Data we Trim Off.
        self.trimmedData = []
    """Calculations"""
    #Calculates Sample Geometry
    def _sampleGeometry(self):
        #Seeing if we're dealing with Tensile or Compression data
        if 'Width (mm)' in list(self.sampleParams.keys()):
            #Converting Thickness and Width into meters
            self.thickness = self.sampleParams['Thickness (mm)']/1000.
            self.width = self.sampleParams['Width (mm)']/1000.
            #Calculating our Area
            self.area = self.thickness * self.width
            #The Gap between Sample Arms
            self.gap = self.sampleParams['Gap (mm)']/1000.
        if 'Diameter (mm)' in list(self.sampleParams.keys()):
            #Converting Diameter
            self.width = self.sampleParams['Diameter (mm)']/1000.
            self.thickness = self.sampleParams['Gap (mm)']/1000.
            self.area = math.pi * (self.width/2)**2
            self.gap = self.thickness
    #Calculate Engineering Values
    def _calculateEngineering(self):
        #Calculating Stress/Strain
        for timePoint in self.stressData:
            #Engineering Stress Calc
            timePoint['Engineering Stress'] = (timePoint['Load']/self.area)/1000000.
            #Engineering Strain
            timePoint['Engineering Strain'] = timePoint['Extension']/self.gap*1.
    #Runs Calculations on the Data
    def _physicalCharacteristics(self):
        self._calculateEngineering()
        #Finding our Maximums.
        self.maxStress = max([i['Engineering Stress'] for i in self.stressData])
        self.maxLoad = max([i['Load'] for i in self.stressData])
        self.maxStrain = max([i['Engineering Strain'] for i in self.stressData])
        self.maxExtension = max([i['Extension'] for i in self.stressData])
    #Crops the Data based on the Major Slope in the First 30% of the Dataset.
    def _cropNew(self):
        searchArea = 0.3 #Determines how much of the Curve we're looking for the dominant slope in
        minimumSlope = 0.1 #Minimum slope to consider "increasing"
        percentageDeviation = 0.7 #Gives the Acceptable Deviation from the Median Slope
        """Cropping the Trailing Bit of the Curve"""
        #Finding the breaking point and trimming everything significantly past that.
        cropPoint = self._findFailure()
        """Cropping the Leading Edge of the Curve"""
        #Looking at the first {searchArea}% of our Data
        toSearch = self.stressData[:int(searchArea*len(self.stressData))]
        #Getting the Slope of our Stress/Strain
        dSS = self._calculateSlopes([i['Engineering Strain'] for i in toSearch],[i['Engineering Stress'] for i in toSearch])
        #Finding the Most Common Slope and all slopes within the Deviation of it.
        dSS = [i for i,j in enumerate(dSS) if j > minimumSlope]
        #Finding the largest consecutive area.
        consecutiveRanges = self._returnRanges(dSS)
        startOfRange = [min(i) for i in consecutiveRanges if max([len(j) for j in consecutiveRanges]) == len(i)]
        if startOfRange:
            startOfRange = startOfRange[0]
        #If there's no starting area, don't bother clipping the start.
        else:
            startOfRange = 0
        #Bg Subtracting
        loadOffset = self.stressData[startOfRange]['Load']
        extensionOffset = self.stressData[startOfRange]['Extension']
        for timePoint in self.stressData:
            if timePoint != self.stressData[startOfRange]:
                timePoint['Load'] = timePoint['Load'] - loadOffset
                timePoint['Extension'] = timePoint['Extension'] - extensionOffset
        #Setting the Originals to 0
        self.stressData[startOfRange]['Load'] = 0
        self.stressData[startOfRange]['Extension'] = 0
        #Calculating our Engineering Values again
        self._calculateEngineering()
        #Clipping the Data
        self.trimmedData.append(self.stressData[:startOfRange+1])
        self.trimmedData.append(self.stressData[cropPoint-1:])
        self.stressData = self.stressData[startOfRange:cropPoint]
        #Recalculating the Parameters
        self._physicalCharacteristics()
        self._calculateYoungs()
        self._calculateToughness()
    #Finds the Failure Point in the Sample
    def _findFailure(self):
        debug = False
        slopeChange = 2 #Amount change in Slope to Trigger Peak Finding
        percentageOfMax = .3 #What percentage of the highest peak can we look for a lower one in
        lookAhead = 3 #How many negative slopes we need after our dip
        percentDrop = 0.005 #Size of Drop
        #Getting the Slope of our Stress/Strain
        dS = self._calculateSlopes([i['Engineering Strain'] for i in self.stressData],[i['Engineering Stress'] for i in self.stressData])
        #Getting the 2nd Derivative
        dSS = self._calculateSlopes([i['Engineering Strain'] for i in self.stressData][1:],dS)
        potentialPeaks = [[i+1,j] for i,j in enumerate(dSS) if j < -1 * slopeChange]
        #Making sure we're decling afterwards
        potentialPeaks = [[i,j] for i,j in potentialPeaks if min([x['Engineering Stress'] for x in self.stressData[i:i+lookAhead+1]]) < self.stressData[i]['Engineering Stress'] * (1-percentDrop)]
        if debug:
            for index,derriv in potentialPeaks:
                stress = self.stressData[index]['Engineering Stress']
                strain = self.stressData[index]['Engineering Strain']
                print('Stress ' + str(round(stress,2)) + ' Strain ' + str(round(strain,2)) + ' Index ' + str(index) + ' dSS ' + str(round(derriv,2)) + ' dS ' + str(round(dS[index],2)))
        if potentialPeaks:
            potentialPeaks = sorted(potentialPeaks, key = lambda x:x[0])
            #Correcting for offset from Derrivs
            return potentialPeaks[0][0] + 1
        #No need to Trim
        else:
            return len(self.stressData) + 1
    #Normalizes the Data by Cropping before the Increase
    def _cropOld(self):
        """Defining our Windows that we're using to Crop"""
        lookPast = 0.02 #Determines how far past the end point we're willing to go.
        minCorrection = 1.4 #x Multiplier from "Flat" slope to "Angled" to determine when to subtract or not
        percentageRange = 0.3 #Percent.age Variation for slope to be "Okay"
        maxCrop = 0.2 # Float - Percentage of the Curve that can be cropped.
                        #Int - Maximum Strain that can be cropped (Not Implemented yet)
        searchArea = 0.3
        if self.stressData:
            #Setting our Sample to report Cropped
            self.cropped = True
            """Cropping the Trailing Bit of the Curve"""
            #Finding the breaking point and trimming everything significantly past that.
            endPoint = [i['Engineering Strain'] for i in self.stressData].index(self.maxStrain)
            cropPoint = int(endPoint + len(self.stressData)*lookPast)
            if cropPoint < len(self.stressData):
                self.stressData = self.stressData[:cropPoint]
            """Cropping the Leading Edge of the Curve"""
            #Looking at the first 50% of our Data
            toSearch = self.stressData[:int(searchArea*len(self.stressData))]
            
            #Getting the Slope of our Stress/Strain
            dSS = self._calculateSlopes([i['Engineering Strain'] for i in toSearch],[i['Engineering Stress'] for i in toSearch])
            #Finding all values within 50% of the Median Slope
            mostCommon = statistics.median(dSS)
            similarToCommon = [i for i,j in enumerate(dSS) if j > mostCommon * percentageRange and j < mostCommon * (1+percentageRange)]
            #Returning if we have nothing to crop
            if len(dSS) == len(similarToCommon) + 1:
                print(str(self.name) + ' - No leading data to Crop')
            #Finding the Next Most Common Slope
            secondCommon = statistics.median([j for i,j in enumerate(dSS) if i not in similarToCommon])
            similiarToSecond = [i for i,j in enumerate(dSS) if j > secondCommon * percentageRange and j < secondCommon * (1+percentageRange)]
            #Looking for which is Greater Slope, proceeding with that
            if secondCommon > mostCommon:
                similarToCommon = similiarToSecond
            #Seeing if we need to Correct or Not
            if minCorrection < mostCommon/secondCommon or minCorrection < secondCommon/mostCommon:
                #Finding the earliest consecutive value within 50% of the slope.
                ranges = self._returnRanges(similarToCommon)
                longestRange = [i for i in ranges if len(i) == max([len(i) for i in ranges])][0]
                startOfRange = min(longestRange)
                #Making sure we're not cropping too much
                if type(maxCrop) == float:
                    #If its a Float, use a Percentage of Length Basis
                    if startOfRange > len(self.stressData) * maxCrop:
                        print('Croppin too Hard')
                        startOfRange = 0
                #If its an Int, use an Amount of Strain
                if type(maxCrop) == int:
                    fire = 1
            else:
                startOfRange = 0
            #Bg Subtracting
            loadOffset = self.stressData[startOfRange]['Load']
            extensionOffset = self.stressData[startOfRange]['Extension']
            for timePoint in self.stressData:
                if timePoint != self.stressData[startOfRange]:
                    timePoint['Load'] = timePoint['Load'] - loadOffset
                    timePoint['Extension'] = timePoint['Extension'] - extensionOffset
            #Setting the Originals to 0
            self.stressData[startOfRange]['Load'] = 0
            self.stressData[startOfRange]['Extension'] = 0
            #Clipping the Data
            self.stressData = self.stressData[startOfRange:]
        self._physicalCharacteristics()
        self._calculateYoungs()
    #Calculates the Toughness
    def _calculateToughness(self):
        #Calculating the Area under the stress vs strain
        self.toughness = numpy.trapz([i['Engineering Stress'] for i in self.stressData],x=[i['Engineering Strain'] for i in self.stressData]) 
    #Calculates the Youngs Modulus
    def _calculateYoungs(self):
        windowSize = 20 #How many Data Points to Look at
        maxForce = 100 #How much MPa before we're out of our Calculation range
        slopeDeviation = .3 #How much our Slopes can Vary from the Median to be considered in Calculation
        #Finding Data Cutoff for Calculation
        belowCutoff = [i['Engineering Strain'] for i in self.stressData if i['Engineering Strain'] > maxForce]
        if belowCutoff:
            maxSample = [i['Engineering Strain'] for i in self.stressData].index(min(belowCutoff))
        else:
            maxSample = len(self.stressData) - windowSize
        #Shrinking our windowSize if needed
        if maxSample < windowSize * 2:
            windowSize = int(maxSample / 4)
        #Fitting Linear Fits to the Range
        windows = [self.stressData[i:i+windowSize] for i in range(0,maxSample)]
        fitWindows = [self._linFit(x) for x in windows]
        #See if we've got an override in the correction File
        correction = self._modulusCorrection()
        if correction:
            #Finding the Closest Value in Strain to the Cutoff
            a = [i['Engineering Strain'] for i in self.stressData]
            firstPeakIndex = min(range(len(a)), key=lambda i: abs(a[i]-float(correction[0])))
            print('Overrode Peak Detection for Sample F' + str(self.formulation) + 'S' + str(self.name))
        else:
            #Finding the first Peak. Looks where the Fit (R^2) and the Slope approach 0.
            #Any exception looks to see if we can find something that looks like a Peak, if not keep going
            if any([i[0] < 0.1 for i in fitWindows]):
                firstPeakIndex = sorted([[i,j[0]*j[2]] for i,j in enumerate(fitWindows)], key = lambda x:x[1])[0][0] - 20
            else:
                firstPeakIndex = len(fitWindows)
        #Finding the Median Slope in the area before the first Peak
        fitWindows = fitWindows[:firstPeakIndex]
        fitWindows = [[i,j] for i,j in enumerate(fitWindows) if abs(j[2]) > 0.7 and j[0] > 0]
        medianSlope = statistics.median([i[1][0] for i in fitWindows])
        #Finding all within {slopeDeviation} of the Median Slope
        matchingFits = [[i,j] for i,j in fitWindows if j[0] > medianSlope * (1 - slopeDeviation) and j[0] < medianSlope * (1 + slopeDeviation)]
        #Making a Fit to the Area where all the slopes are similar
        finalFitData = self.stressData[min([i[0] for i in matchingFits])+10:max([i[0] for i in matchingFits])+10]
        bestFit = self._linFit(finalFitData)
        #print('\n'.join([str(i[0]) + ' - ' + str(i[2]) for i in fitWindows]))
        #windows = sorted(windows, key=lambda x:-1*self._linFit(x)[0])
        #bestFit = self._linFit(windows[0])
        self.youngSlope = bestFit[0]
        sample.youngOffset = bestFit[1]
        xmin = min([i['Engineering Strain'] for i in finalFitData])
        xmax = max([i['Engineering Strain'] for i in finalFitData])
        self.youngLine = [[xmin,xmin*self.youngSlope + self.youngOffset],
                            [xmax, xmax*self.youngSlope + self.youngOffset]]
        self.modulusFit = round(abs(bestFit[2]),2)
    #Loads in the Modulus Correction File and returns the Peak Location if Possible
    def _modulusCorrection(self):
        import csv
        with open('modulusCorrection.csv') as inFile:
            reader = csv.reader(inFile)
            header = next(reader)
            lines = [i for i in reader]
        return [i[2] for i in lines if int(i[0]) == self.name and int(i[1]) == self.formulation]
    """Small Auxilary Functions"""
    #Generates a Value Dictionary based on a Category
    def _valueDict(self,line,header,category,selection):
        return dict([[j,self._typeVal(i)] for i,j,k in zip(line,header,category) if k == selection])
    #Returns instantaneous Slopes
    def _calculateSlopes(self,X,Y):
        dY = (numpy.roll(Y, -1, axis=0) - Y)[:-1]
        dX = (numpy.roll(X, -1, axis=0) - X)[:-1]
        slopes = dY/dX
        return slopes
    #Returns the Average and CV of an Attribute
    def _calcAvgCv(self,dataset,attributeName):
        data = [i[attributeName] for i in dataset]
        mean = sum(data)*1./len(data)
        stdev = statistics.stdev(data)
        cv = stdev/mean
        return mean,cv
    #Returns Lists of Values that are consecutive
    def _returnRanges(self,data):
        from operator import itemgetter
        from itertools import groupby
        #Returning lists organized by consecutive numbers
        values = []
        for k,g in groupby(enumerate(data), lambda ix : ix[0] - ix[1]):
            values.append(list(map(itemgetter(1), g)))
        return values
    #Returns the x Value at which the Slope changes the Most
    def _maxChange(self,x,y):
        firstDiff = self._calculateSlopes(x,y)
        #Seeing where the Maximum negative change is, and returning that X value
        return x[numpy.where(firstDiff == min(firstDiff))[0][0]]
    #Returns a Linear Fit of a Stress/Strain Dataset
    def _linFit(self,dataset):
        from scipy import stats
        return stats.linregress([i['Engineering Strain'] for i in dataset],[j['Engineering Stress'] for j in dataset])
    #Attempts to turn the String coming in into a better Type
    def _typeVal(self,val):
        try:
            return int(val)
        except:
            try:
                return float(val)
            except:
                if val.lower() == 'true':
                    return True
                elif val.lower() == 'false':
                    return False
                else:
                    return val
    """Debug Functions"""
    def debug_calcSlope(self,strain):
        lookAround = 10
        approximate = 0.1
        slopeChange = 2
        lookAhead = 3
        #Finding all Values within 10% of the given
        potentialValues = [i for i,j in enumerate(self.stressData) if j['Engineering Strain'] < strain * (1+ approximate) and j['Engineering Strain'] > strain * (1 - approximate)]
        index = potentialValues[math.floor(len(potentialValues)/2)]
        stressData = self.stressData[index-10:index+10]
        #Getting the Slope of our Stress/Strain
        dS = self._calculateSlopes([i['Engineering Strain'] for i in stressData],[i['Engineering Stress'] for i in stressData])
        #Getting the 2nd Derivative
        dSS = self._calculateSlopes([i['Engineering Strain'] for i in stressData][1:],dS)
        potentialPeaks = [[i+1,j] for i,j in enumerate(dSS) if j < (-1 * slopeChange)]
        lookaheadPeaks = [[[x for x in dS[i:i+lookAhead+1]],j] for i,j in potentialPeaks]
        print('Debug Slopes')
        print(self.stressData[index]['Engineering Strain'])
        print(dS)
        print(dSS)
        print(potentialPeaks)
        print(lookaheadPeaks)

#Parses out an Instron generated Raw Data file
#Returns information about Stress/Strain
class stressDataFile():
    def __init__(self,fileLoc):
        self.data = []
        self.loadFile(fileLoc)
    #Loads in the Dataset
    def loadFile(self,loc):
        with open(loc,'r') as inFile:
            data = inFile.read()
        #Splitting up the CSV
        lines = [i.split(',') for i in data.split('\n') if i]
        #Seeing if we've got a date header
        startInt = [i for i,val in enumerate(lines) if 'Time' in val[0]][0]
        lines = lines[startInt:]
        header = lines.pop(0)
        units = lines.pop(0)
        for line in lines:
            #Making a Dictionary out of the values
            lineDict = {}
            for name,val in zip(header,line):
                val = val.replace('"','')
                lineDict[name] = self._type(val)
                if name == 'Extension':
                    lineDict[name] == lineDict[name]/1000.
            self.data.append(lineDict)
    #Typing the data coming in
    def _type(self,string):
        try:
            return int(string)
        except:
            try:
                return float(string)
            except:
                return string

#Parses in the proposedSamples file from TS_EMO
class proposedSamples():
    def __init__(self,fileLoc):
        with open(fileLoc,'r') as inFile:
            lines = inFile.read()
            lines = lines.split('\n')
            #Finding the Breaks between Sections
            breaks = [i for i,j in enumerate(lines) if j == '']
            #Finding the Compositions
            self.compositions = self._parseTabs(lines[breaks[0]+3:breaks[1]])
            #Expected Performance
            self.performance = self._parseTabs(lines[breaks[1]+3:breaks[2]])
            #Finding the Hypervolume Improvement
            self.hypervolume = float(lines[breaks[-1]+1:][0].split(' ')[-1])
            self.lines = [proposedSample(i,j,self.hypervolume) for i,j in zip(self.compositions,self.performance)]
            
    #Parses Composition Lines
    def _parseTabs(self,lines):
        return [[float(x.strip()) for x in i.split(' ') if x != ''] for i in lines]

class proposedSample():
    def __init__(self,composition,performance,hypervolume):
        self.composition = composition
        self.performance = performance
        self.hypervolume = hypervolume

"""Plotting Classes"""
#Base Plotting Class.
#Used for Consistent Styling
class plot():
    def __init__(self,inputData,show):
        self.inputData = inputData
        #Plot Styling
        plt.style.use('ggplot')
        #Styling for Text Boxes
        self.props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        #Making a Directory for Plots if we're Saving Out
        self.show = show
        if self.show == False:
            self._checkDirectory()
    #Changing the Colors
    def _setColors(self):
        colors = [['b',[i for i in range(23,52)]+[i for i in range(57,62)]+[75,76]],
                    ['g',[43,44,46,75,76]]]
        for color,formulations in colors:
            for i in self.inputData:
                if i.name in formulations:
                    i.color = color

    
#Making our Stress/Strain Plot
class stressStrainPlot(plot):
    def __init__(self,inputData,show = True, overwrite = False):
        plot.__init__(self,inputData,show)
        #Triggers whether or not we Overwrite preexisting plots
        self.overwrite = overwrite
        #Generating the Plot.
        if self.inputData.stressData:
            self.fig, self.ax = plt.subplots()
            self._plotData()
            self._plotMaximums()
            self._plotModulus()
            self._plotTrimmed()
            self._axisLabels()
            self._performanceInfo()
            self._compositionInfo()
            self._displayChart()
            plt.close()
    #Plots the Data
    def _plotData(self):
        self.ax.plot([i['Engineering Strain'] for i in self.inputData.stressData],[i['Engineering Stress'] for i in self.inputData.stressData])
        plt.fill_between([i['Engineering Strain'] for i in self.inputData.stressData],[i['Engineering Stress'] for i in self.inputData.stressData],alpha=0.3)
    #Dotted Lines for Max Stress/Strain
    def _plotMaximums(self):
        self.ax.vlines(self.inputData.maxStrain, 0, self.inputData.maxStress, linestyle="dashed")
        self.ax.hlines(self.inputData.maxStress, 0, self.inputData.maxStrain, linestyle="dashed")
    #Plots the Modulus
    def _plotModulus(self):
        if hasattr(self.inputData,"youngLine"):
            self.performanceStr = f"Modulus {round(self.inputData.youngSlope,2)}"
            #Plotting the Line Segment
            self.ax.plot([i[0] for i in self.inputData.youngLine],[i[1] for i in self.inputData.youngLine],c='g')
        else:
            self.performanceStr = ''
    #Plots out our Trimmed Data
    def _plotTrimmed(self):
        if hasattr(self.inputData,'trimmedData'):
            for stressData in self.inputData.trimmedData:
                self.ax.plot([i['Engineering Strain'] for i in stressData],[i['Engineering Stress'] for i in stressData],c='b')
    #Making a Little Chart with Performance Metrics
    def _performanceInfo(self):
        #Calculating our Performance String to go in the Box
        computedStr = f"\nmax Stress {round(self.inputData.maxStress,2)} MPa \nmax Strain {round(self.inputData.maxStrain,2)} \nToughness {round(self.inputData.toughness,2)}"
        computedStr = self.performanceStr + computedStr
        #Plotting the Text Box
        self.ax.text(0.6, 0.15, computedStr, transform=self.ax.transAxes, fontsize=10,verticalalignment='top', bbox=self.props)
    #Making a Chart with Composition Information
    def _compositionInfo(self):
        computedStr = '\n'.join([key + '-' + str(round(self.inputData.composition[key]*100,1)) + '%' for key in self.inputData.composition.keys()])
        self.ax.text(0.05, 0.9, 'Formulation\n' + computedStr, transform=self.ax.transAxes, fontsize=10,verticalalignment='top', bbox=self.props)
    #Making our Labels on Axis
    def _axisLabels(self):
        if hasattr(self.inputData,'name'):
            if hasattr(self.inputData, 'formulation'):
                title = 'Formulation ' + str(self.inputData.formulation) + ' Sample ' + str(self.inputData.name)
            else:
                title = 'Sample ' + str(self.inputData.name)
        else:
            title = 'Stress Strain'
        plt.title(title)
        self.ax.set_ylabel('Engineering Stress (MPa)')
        self.ax.set_xlabel('Engineering Strain (-)')  
    #Making our Plot Folder if we don't have it
    def _checkDirectory(self):
        if not os.path.exists(os.path.join('Plots','StressStrain')):
            os.makedirs(os.path.join('Plots','StressStrain'))
    def _displayChart(self):
        if self.show:
            plt.show()
        else:
            #Deciding what to call the Sample.
            if hasattr(self.inputData,'formulation') and self.inputData.formulation:
                sampleName = 'F'+str(self.inputData.formulation)+'S'+str(self.inputData.name)+'.png'
            else:
                sampleName = str(self.inputData.name)+'.png'
            #Making sure we need to make the plot
            if self.overwrite or sampleName not in os.listdir(os.path.join('Plots','StressStrain')):
                #Saving the Plot Somewhere Useful
                savePath = os.path.join('Plots','StressStrain',sampleName)
                plt.savefig(savePath, dpi=400)
                #Clearing it
                plt.clf()
                #Update Message to say We're done Plotting
                print('Plotted - ' + sampleName)

#Makes a Plot of Performance Space
class performanceSpacePlot(plot):
    def __init__(self,inputData,show = False):
        plot.__init__(self,inputData,show)
        #Assign Colors
        #self._assignColors()
        self._setColors()
        #Determine the plots we need to generate
        self.metrics = {'maxStrain':'Compression At Failure','maxStress':'Compression Strength','youngSlope':'Compression Modulus','toughness':'Toughness'}
        plots = [[i,j] for i,j in zip(list(self.metrics.keys()),list(self.metrics.keys())[1:]+[list(self.metrics.keys())[0]])]
        plots.append(['maxStress','toughness'])
        for attr1,attr2 in plots:
            self._makePlot(attr1,attr2)
            self._addLabels(attr1,attr2)
            self._displayChart(attr1,attr2)
    #Plots the Values
    def _makePlot(self,attr1,attr2):
        self.fig, self.ax = plt.subplots()
        self.legend = []
        self.legendColors = []
        for n in self.inputData:
            #Making sure we've got data to plot
            if hasattr(n,attr1) and hasattr(n,attr2) and getattr(n,attr1) and getattr(n,attr2):
                #Plotting Average Values
                x = getattr(n,attr1)
                y = getattr(n,attr2)
                a = self.ax.scatter(x,y, c=[n.color], label=n.name)
                self.legend.append(a)
                #Plotting Error Bars
                #markers, caps, bars = plt.errorbar(x,y,yerr=getattr(n,attr2+'CV')*y,xerr=getattr(n,attr1+'CV')*x,ecolor=n.color)
                #Setting our Alpha for Error Bars
                #[bar.set_alpha(0.5) for bar in bars]
                #[cap.set_alpha(0.5) for cap in caps]
            #for x,y in zip([getattr(i,attr1) for i in n.samples if hasattr(i,attr1)],[getattr(i,attr2) for i in n.samples if hasattr(i,attr2)]):
            #    a = self.ax.scatter(x,y, c=[n.color], label=n.name, alpha=0.5)
            #    if n.color not in self.legendColors:
            #        self.legendColors.append(n.color)
            #        self.legend.append(a)
    #Makes the Labels
    def _addLabels(self,attr1,attr2):
        #General Plotting Niceness
        plt.title("Performance Space")
        # Shrink current axis by 20%
        #box = self.ax.get_position()
        #self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        #ncol = math.ceil(len(self.legend)/20)
        #self.ax.legend(handles=self.legend, loc='center left', bbox_to_anchor=(1, 0.5), ncol=ncol)
        #Converting to the Display Names
        self.ax.set_ylabel(self.metrics[attr2])
        self.ax.set_xlabel(self.metrics[attr1])
    #Saves the Plot
    def _displayChart(self,attr1,attr2):
        if self.show:
            plt.show()
        else:
            #Saving the Plot Somewhere Useful
            plt.savefig(os.path.join('Plots','PerformanceSpace',self.metrics[attr1]+"_"+self.metrics[attr2]+'.png'), dpi=600)
            #Clearing it
            plt.clf()
            print('Saved '+self.metrics[attr1]+" vs "+self.metrics[attr2]+' plot')
        plt.close()
    #Assigns Colors to Samples or Formulations
    def _assignColors(self):
        import matplotlib._color_data as mcd
        import matplotlib.cm as cm
        colors = cm.rainbow(numpy.linspace(0, 1, len(self.inputData)))
        for obj,color in zip(self.inputData,colors):
            obj.color = color
    #Making our Plot Folder if we don't have it
    def _checkDirectory(self):
        if not os.path.exists(os.path.join('Plots','PerformanceSpace')):
            os.makedirs(os.path.join('Plots','PerformanceSpace'))

#Making a 3d Plot
class plot3d(plot):
    def __init__(self,inputData,show = True):
        plot.__init__(self,inputData,show)
        #Setting up our 3d Plot
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        self._addLabels()
        self._setColors()
        self._plotData()
        self._invertAxis()
        self._displayChart()
    #Inverts the Axis
    def _invertAxis(self):
        self.ax.set_xlim(plt.xlim()[-1],plt.xlim()[0])
        self.ax.set_ylim(plt.ylim()[-1],plt.ylim()[0])
        #self.ax.set_zlim(plt.zlim()[-1],plt.zlim()[0])
        
    #Makes the Labels
    def _addLabels(self):
        #General Plotting Niceness
        plt.title("Performance Space")
        # Shrink current axis by 20%
        #box = self.ax.get_position()
        #self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        #ncol = math.ceil(len(self.legend)/20)
        #self.ax.legend(handles=self.legend, loc='center left', bbox_to_anchor=(t1, 0.5), ncol=ncol)
        #Converting to the Display Names
        self.ax.set_xlabel('Compression Strength')
        self.ax.set_ylabel('Compression Modulus')
        self.ax.set_zlabel('Toughness')

    #Plotting the Data
    def _plotData(self):
        dataToFetch = ['maxStress','youngSlope','toughness','color']
        self.ax.scatter3D([getattr(i,dataToFetch[0]) for i in self.inputData if hasattr(i,dataToFetch[0]) and getattr(i,dataToFetch[0])], 
                            [getattr(i,dataToFetch[1]) for i in self.inputData if hasattr(i,dataToFetch[0]) and getattr(i,dataToFetch[0])],
                            [getattr(i,dataToFetch[2]) for i in self.inputData if hasattr(i,dataToFetch[0]) and getattr(i,dataToFetch[0])],
                            c=[getattr(i,dataToFetch[3]) for i in self.inputData if hasattr(i,dataToFetch[0]) and getattr(i,dataToFetch[0])])
    #Making our Plot Folder if we don't have it
    def _checkDirectory(self):
        if not os.path.exists(os.path.join('Plots','PerformanceSpace')):
            os.makedirs(os.path.join('Plots','PerformanceSpace'))
    #Saves the Plot
    def _displayChart(self):
        if self.show:
            plt.show()
        else:
            #Saving the Plot Somewhere Useful
            plt.savefig(os.path.join('Plots','PerformanceSpace','3d.png'), dpi=600)
            #Clearing it
            plt.clf()
            print('Saved 3d plot')
        plt.close()

if __name__ == "__main__":
    test = dataStruct(processAll = True)
    #print(test.make_dataframe())
    test.saveFormulations()
    
    performanceSpacePlot(test.formulations, show = False)
    for sample in test.samples:
        stressStrainPlot(sample, show = False)

    #plot3d(test.formulations, show = True)
    #a = [i for i in test.samples if i.formulation == 103]
    #for x in a:
        #stressStrainPlot(x, show = True)

    #import os
    #proposedSamples(os.path.join('Plots','TSEmoOpt','3','proposedSamples_1.txt'))