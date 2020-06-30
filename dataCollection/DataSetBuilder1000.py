import praw
import pprint
import string
import re
import os
import csv
from prawcore.exceptions import Forbidden, NotFound
from SlangList import SlangList

def crawData(subreddit, drugList, inputFile1, inputFile2, inputFile3, initNumUsers, numWantedUsers, startFlag, nameSet, searchRange) :
    fileName = inputFile1
    fileName2 = inputFile2
    fileName3 = inputFile3

    if startFlag == True:
        if os.path.exists(fileName):
            os.remove(fileName)
        else:
            print("The fileTemp does not exist")
        if os.path.exists(fileName2):
            os.remove(fileName2)
        else:
            print("The file2 does not exist")
        if os.path.exists(fileName3):
            os.remove(fileName3)
        else:
            print("The file3Temp does not exist")
    else:
        pass

    totalNumUsers = initNumUsers + 1

    header = ['student1', 'student2', 'consensus', 'userID', 'content', 'keywords']
    file = open(fileName, 'a')
    writer = csv.writer(file)
    file3 = open(fileName3, 'a')
    writer3 = csv.writer(file3)
    if startFlag == True:
        writer.writerow(header)
        writer3.writerow(header)
    file2 = open(fileName2, 'a')


    #for submission in subreddit.hot(limit = None):
    for submission in subreddit.top(searchRange, limit = None):
    #for submission in subreddit.submissions(1359566260, 1391102260):  #from 2019-11-01 to 2019-12-31

        redditor = submission.author
        if redditor == None:
            continue
        if redditor.name not in nameSet:
            nameSet.add(redditor.name)
        else:
            continue
        flag = 0   # indicate that there is no comments in terms of "drugs" of this user

        keyWordList = []
        try:
            for comment in redditor.comments.top('month'):
                try:
                    commentStr = comment.body  # get the comment body
                    commentSplit = commentStr.split(" ")
                    if len(commentSplit) < 15:
                        continue
                except (Forbidden, NotFound) as err:
                    continue

                #data preprossessing
                table = commentStr.maketrans("", "", string.punctuation)
                noPunctComment = commentStr.translate(table)  # remove the punctuation
                newComment = noPunctComment.lower()  # convert to lowercase letter

                num = 0
                prenum = 0

                for i in range(len(drugList)):
                    # num = num + origStr.count(origList[i])
                    # the same as count. But can avoid counting substring in non-complete words
                    # e.g. count("dog" in "love dogs") = 0
                    num = num + sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(drugList[i]), newComment))
                    # show the street words which appear in the sentence (not complete!).
                    if prenum != num:
                        if flag == 0:
                            file.write(",,," + redditor.name + ",")
                            file3.write(",,," + redditor.name + "," +'"')
                        else:
                            pass

                        # find the non-duplicate keyWord list
                        flagTemp = False # drugList[i] in keyWordList
                        for keyWord in keyWordList:
                            if drugList[i] == keyWord:
                                flagTemp = True
                        if flagTemp == False:
                            keyWordList.append(drugList[i])

                        file.write(str(newComment).replace('\n', ' '))
                        file2.write(str(totalNumUsers) + '^'  + redditor.name + '^' + str(drugList[i]) + '^' + str(commentStr) + "\n" + "\n")
                        file3.write(str(commentStr).replace('\n', ' ').replace('"', '') + '\n') # double quotes will affect the csv split. So delete internal double quotes here.
                        flag = 1
                        break
                    prenum = num

            if flag == 1:
                # write the key word at the end
                file.write(",")
                file3.write('"' + ",")
                for j in range(len(keyWordList) - 1):
                    file.write(keyWordList[j] + "^")
                    file3.write(keyWordList[j] + "^")
                file.write(keyWordList[len(keyWordList) - 1])
                file3.write(keyWordList[len(keyWordList) - 1])

                print(totalNumUsers)
                totalNumUsers += 1
                file.write("\n")
                file3.write("\n")
                file2.write("--------------------------------------------------------------------------------------------------------------\n")
            if totalNumUsers == numWantedUsers + 1:
                break
        except (Forbidden, NotFound) as err:
            continue
    file.close()
    file2.close()
    file3.close()
    print("The total numbers of users is: " + str(totalNumUsers - 1))
    return totalNumUsers - 1

def reOrderFile(fileIn, fileOut): #switch two columns: content and keywords
    if os.path.exists(fileOut):
        os.remove(fileOut)
    else:
        print("The file does not exist")

    with open(fileIn, 'r') as infile, open(fileOut, 'a') as outfile:
        fieldnames = ['student1', 'student2', 'consensus', 'userID', 'keywords', 'content']
        writer = csv.DictWriter(outfile, fieldnames = fieldnames, extrasaction = 'ignore')
        writer.writeheader()
        for row in csv.DictReader(infile):
            writer.writerow(row)


def main():
    reddit = praw.Reddit(client_id='kb8Ae0P9oQ8UOg',
                         client_secret='MYc6XcLBG-Zhu_9mTh9OWPw7Jrc',
                         user_agent='CWRU research by /u/wycuestc',
                         username='wycuestc',
                         password='Wyc1301042111')
    subreddit = reddit.subreddit('opiates+OpiatesRecovery+Drugs')
    # subreddit = reddit.subreddit('Drugs')

    # get input list from SlangList.py
    drugType = 'opium'
    sl = SlangList(drugType)
    drugList = sl.getDrugList(drugType)
    drugStreetList = sl.getDrugStreetList(drugType)

    # the total number of users we want to get
    initNumUsers = 0
    numWantedUsers = 3000
    nameSet = set()

    scriptDir = os.path.dirname(__file__)
    fileNameTemp = os.path.join(scriptDir, "dataset1000_building/dataOpiumStreetTemp.csv")
    fileName1 = os.path.join(scriptDir, "dataset1000_building/dataOpiumStreet.csv")
    fileName2 = os.path.join(scriptDir, "dataset1000_building/dataReadOpiumStreet.txt")
    fileName3Temp = os.path.join(scriptDir, "dataset1000_building/dataOriginOpiumStreetTemp.csv")
    fileName3 = os.path.join(scriptDir, "dataset1000_building/dataOriginOpiumStreet.csv")

    initNumUsers = crawData(subreddit, drugList + drugStreetList, fileNameTemp, fileName2, fileName3Temp, initNumUsers, numWantedUsers, True, nameSet, 'week')
    input("Press Enter to continue...")
    print("now the initNumUsers is " + str(initNumUsers))
    initNumUsers = crawData(subreddit, drugList + drugStreetList, fileNameTemp, fileName2, fileName3Temp, initNumUsers, numWantedUsers, False, nameSet, 'month')
    input("Press Enter to continue...")
    print("now the initNumUsers is " + str(initNumUsers))
    crawData(subreddit, drugList + drugStreetList, fileNameTemp, fileName2, fileName3Temp, initNumUsers, numWantedUsers, False, nameSet, 'year')
    reOrderFile(fileNameTemp, fileName1)
    reOrderFile(fileName3Temp, fileName3)

if __name__ == '__main__':
    main()