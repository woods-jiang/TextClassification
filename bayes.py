'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *
import os
                 
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

def loadDataSet():
    
    fileNames = getTrainFileName()
    docList=[]; classList = []; priorProbabilities=[]
    
    
    
    # train set
    for i in range(10):
        for fileName in fileNames[i]:
            wordList = textParse(open(fileName).read())
            docList.append(wordList)
            classList.append(i)
        
    # prior probalibities
    for i in range(10):
        priorProbabilities.append(len(docList[i])/len(classList))
        
    vocabList = createVocabList(docList)    #create vocabulary
    
    trainMat=[]; trainClasses=classList
    
    for doc in docList:
        trainMat.append(bagOfWords2VecMN(vocabList, doc))
    
    # training
    pVect = trainNB0(array(trainMat),array(trainClasses))

    #test set
    fileNames = getTestFileName()
    docList=[]; classList = []
    
    for i in range(10):
        for fileName in fileNames[i]:
            wordList = textParse(open(fileName).read())
            docList.append(wordList)
            classList.append(i)
    
    testMat=[]; testClasses=classList
    
    for doc in docList:
        testMat.append(bagOfWords2VecMN(vocabList, doc))
    
    # test
    errorCount = 0
    for i in range(shape(testMat)):
        if classifyNB(array(testMat[i]),pVect,priorProbabilities) != testClasses[i]:
            errorCount += 1

    print 'the error rate is: ',float(errorCount)/len(testSet)
    
    return trainMatrix,trainCategory

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    pNum = ones((10,numWords))
    pDenom = ones(10)*10
    for i in range(numTrainDocs):
        pNum[trainCategory[i]] += trainMatrix[i]
        pDenom[trainCategory[i]] += sum(trainMatrix[i])
            
    pVect = []
    for i in range(10):
        pVect.append(log(pNum[i]/pDenom[i]))
    return pVect

def classifyNB(vec2Classify, pVec, pClass1):
#    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
#    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
#    if p1 > p0:
#        return 1
#    else: 
#        return 0
    pass;
    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
    
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = range(50); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)
    #return vocabList,fullText

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]

def getTrainFileName():
    
    fileNames = []
    
    for trainDir in os.listdir('./training'):
        oneClassFileName = []
        for fileName in os.listdir('./training/' + trainDir):
            oneClassFileName.append('./training/' + trainDir + '/' + fileName)
        fileNames.append(oneClassFileName)
        
    return fileNames


def getTestFileName():
    
    fileNames = []
    
    for testDir in os.listdir('./test'):
        oneClassFileName = []
        for fileName in os.listdir('./training/' + testDir):
            oneClassFileName.append('./training/' + testDir + '/' + fileName)
        fileNames.append(oneClassFileName)
        
    return fileNames
    
    