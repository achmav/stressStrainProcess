#Parses out Shimadzu Data into Instron Format.
import csv
import os
import datetime


#Returns a Date String of when the File was last Modified
def parseModified(fileLoc):
    timeSinceEpoch = os.path.getmtime(fileLoc)
    dateTimeObj = datetime.datetime.fromtimestamp(timeSinceEpoch)
    return dateTimeObj.strftime('%m/%d/%Y')

#Reformulates into F###S# Format
def sampleName(fileName,shimadzuName):
    #Getting the Formulation Number from the Sample Name
    fNum = fileName.split('.')[0].split(os.path.sep)[-1]
    #Getting the Replicate Number
    sNum = shimadzuName.split('_')[-1].strip()
    #Formulating into our Final name
    return 'F' + fNum + 'S' + sNum

def parseShimadzu(fileLoc,saveFolder = 'Test'):
    with open(fileLoc,'r') as inFile:
        readFile = csv.reader(inFile)
        lines = [i for i in readFile]
        #Finding the Samples within the Individual File
        samples = [[c,v] for c,v in enumerate(lines[0]) if v != '']
        #Iterating through the Samples
        for index,sample in samples:
            #Writing out the New File
            with open(os.path.join(saveFolder,sampleName(fileLoc,sample)+'.csv'),'w') as outFile:
                writer = csv.writer(outFile,delimiter=',', lineterminator='\n')
                #Writing out Headers
                writer.writerow(['General : Start date',parseModified(fileLoc)])
                writer.writerow([])
                writer.writerow(['Time','Extension','Load'])
                writer.writerow(['(s)','(mm)','(N)'])
                #Transposing the Data
                timeData = [i[index] for i in lines[3:]]
                forceData = [i[index+1] for i in lines[3:]]
                strokeData = [i[index+2] for i in lines[3:]]
                for row in zip(timeData,strokeData,forceData):
                    writer.writerow(row)

#Returns a list of Files that are of Shimadzu Origin for Compression Data
def getShimadzuFiles(fileDir):
    files = [os.path.join(fileDir,i) for i in os.listdir(fileDir) if '.csv' in i if shimadzuData(os.path.join(fileDir,i))]
    return files

#Returns True if 
def shimadzuData(fileLoc):
    with open(fileLoc,'r') as inFile:
        readFile = csv.reader(inFile)
        lines = [i for i in readFile]
        return lines[1][0] == 'Time' and lines[2][0] == 'sec'

if __name__ == "__main__":
    for shiFile in getShimadzuFiles('Shimadzu'):
        print('Parsing - ' + shiFile)
        parseShimadzu(shiFile)
    
    pass
