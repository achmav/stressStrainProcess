import os
import shutil

#Finding all of our Folders we could have samples in
#Sorting via the Formulation Number
subFolders = [x[0] for x in os.walk('.') if x[0] != '.' and 'RawData' in x[0]]
#Making our Data Folder if we don't have it
if 'DataNew' not in os.listdir('.'):
    os.mkdir('DataNew')

#Starting to move Data Around
for folder in subFolders:
    #Formulation Number
    #formulationNumber = int(folder.split('F')[-1].split('.')[0])
    print(folder)
    formulationNumber = int(os.path.split(folder)[-1].split('.')[0])
    #Finding the Sample Data in each subFolder
    files = [i for i in os.listdir(folder) if 'Specimen' in i and '.csv' in i]
    for individual in files:
        sampleNumber = int(individual.split('_')[-1].split('.')[0])
        sampleName = 'F'+str(formulationNumber)+'S'+str(sampleNumber)+'.csv'
        #Making sure we haven't moved this Sample yet Already
        if sampleName not in os.listdir('DataNew'):
            print('Moved Sample - ' + sampleName)
            shutil.copy(os.path.join(folder,individual),os.path.join('DataNew',sampleName))

""" Old Methodology for Renaming    
#Starting to move Data Around
for folder in subFolders:
    #Finding the Sample Data in each subFolder
    files = [i for i in os.listdir(folder) if 'Specimen' in i and '.csv' in i]
    files = sorted(files,key=lambda x:int(x.split('_')[-1].split('.')[0]))
    #Going through the Files.
    for individual in files:
        #Finding the Sample Number
        if os.listdir('Data') and any(['Specimen' in x for x in os.listdir('Data')]):
            specimenNum = max([int(x.split('_')[-1].split('.')[0]) for x in os.listdir('Data')]) + 1
        else:
            specimenNum = 1
        #Creating the New Name
        newName = 'Specimen_RawData_' + str(specimenNum) + '.csv'
        shutil.copy(os.path.join(folder,individual),os.path.join('Data',newName))
"""
