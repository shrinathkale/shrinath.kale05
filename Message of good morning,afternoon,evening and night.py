import time
time = time.strftime("%H:%M:%S")
print(time)
if (time >= "0:0:0" and time < "12:0:0"):
    print("Good Morning")
elif (time >= "12:0:0" and time < "17:0:0"):
    print("Good Afternoon")
elif (time >= "17:0:0" and time < "19:0:0"):
    print("Good Evening")
else:
    print("Good Night")