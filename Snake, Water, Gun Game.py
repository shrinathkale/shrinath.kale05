print("WELCOME TO SNAKE WATER GUN GAME")
print("In this game you have to select number for snake or water or gun & computer will also select & finally decide you win or loss")
import random
score = 0
while(True):
    computer = random.randint(-1,1)
    user = int(input("Enter number: \n-1 for snake \n 0 for water \n 1 for gun \n 9 for exit \n"))
    if(user == computer):
        print("Draw")
        print("you: ",user)
        print("computer: ",computer)
        score += 1
    elif(user == -1 and computer == 0):
        print("Win")
        print("you: ",user)
        print("computer ",computer)
        score += 2
    elif(user == -1 and computer == 1):
        print("Lose")
        print("you: ",user)
        print("computer ",computer)
        score += 0
        break
    elif(user == 0 and computer == -1):
        print("win")
        print("you: ",user)
        print("computer ",computer)
        score += 2
    elif(user == 0 and computer == 1):
        print("Win")
        print("you: ",user)
        print("computer ",computer)
        score += 2
    elif(user == 1 and computer == -1):
        print("Win")
        print("you: ",user)
        print("computer ",computer)
        score += 2
    elif(user == 1 and computer == 0):
        print("Lose")
        print("you: ",user)
        print("computer ",computer)
        score += 0
        break
    else:
        print("YOU ARE EXIT")
        break
print("YOUR SCORE IS: ",score)
if(score == 0):
    print("BAD LUCK")
elif(score > 0 and score <= 15):
    print("GOOD")
elif(score < 15 and score <= 30):
    print("GREAT")
else:
    print("EXCELLENT")