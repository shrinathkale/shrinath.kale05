import random
import string
ip = input("Enter any string: ")
code = int(input("Enter 1 for coding and 0 for decoding: "))
word = ip.split(" ")
code = True if(code == 1) else False
if(code):
    for w in word:
        if (len(w)>3):
            char = string.ascii_letters
            first = random.choice(char)
            last = random.choice(char)
            coding = first + w[1:] + w[0] + last
            print("coding is: ",coding,"\n\n")
        else:
            coding = w[::-1]
            print("coding is: ",coding,"\n\n")
else:
    for w in word:
        if (len(w)>3):
            w1 = w[1:-1]
            decoding = w1[-1] + w1[0:-1]
            print("coding is: ",decoding,"\n\n")
        else:
            coding = w[::-1]
            print("coding is: ",coding,"\n\n")
             