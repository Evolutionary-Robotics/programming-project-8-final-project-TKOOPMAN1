import numpy as np
import matplotlib.pyplot as plt
import sys
import os

#expname = sys.argv[1]
#reps = int(sys.argv[2])

currentpath = os.getcwd()
os.chdir(currentpath)

#for i in range(1):
ev = np.load("evol"+str(8)+".npy")
wev = np.load("wevol"+str(8)+".npy")
avev = np.load("avevol"+str(8)+".npy")
plt.plot(ev)
plt.plot(wev)
plt.plot(avev)
#print(i,ev[-1])

plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("Evolution")
plt.show()