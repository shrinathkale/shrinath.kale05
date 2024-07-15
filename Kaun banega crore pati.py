ques = [["which language was used for fb?","python","java","php","none",3],
        ["which language was used for fb?","python","java","php","none",3],
        ["which language was used for fb?","python","java","php","none",3],
        ["which language was used for fb?","python","java","php","none",3],
        ["which language was used for fb?","python","java","php","none",3],
        ["which language was used for fb?","python","java","php","none",3],
        ["which language was used for fb?","python","java","php","none",3],
        ["which language was used for fb?","python","java","php","none",3],
        ["which language was used for fb?","python","java","php","none",3],
        ["which language was used for fb?","python","java","php","none",3],
        ["which language was used for fb?","python","java","php","none",3],
        ["which language was used for fb?","python","java","php","none",3],
        ["which language was used for fb?","python","java","php","none",3]]
levels = [1000,2000,3000,5000,10000,20000,30000,40000,80000,160000,320000,640000,10000000]
money = 0
for i in range(0,len(ques)):
    que = ques[i]
    print("\n\nQuestion for", levels[i], "is:",que[0])
    print("1.",que[1],      "2.",que[2])
    print("3.",que[3],      "4.",que[4])
    reply = int(input("Enter no of correct option (enter 9 for quiting): "))
    if (reply == que[-1]):
        print("Correct!")
        if (i == 4):
            money = 10000
        elif (i == 9):
            money = 320000
        elif (i == 14):
            money = 100000000
    elif(reply == 9):
        print("\n\nYOU HAS QUITED")
        money = levels[i-1]
        break
    else:
        print("\n\nWrong answer")
        break
print("total amount is:",money)