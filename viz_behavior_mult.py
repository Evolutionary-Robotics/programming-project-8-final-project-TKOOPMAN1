import EA                  
import fnn                
import Vehicle           
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

#expname = sys.argv[1]
#id = sys.argv[2]
lights = Vehicle.Light()

# currentpath = os.getcwd()
# os.chdir(currentpath+'/'+expname)

# ANN Params
layers = [2,5,5,2]

# Task Params
duration = 500
stepsize = 0.02    
noisestd = 0.01 

# Time
time = np.arange(0.0,duration,stepsize)

# Load genotype
bestind_genotype = np.load("gen1.npy")
worstind_genotype = np.load("wgen1.npy")

angles = 35
def evaluate(genotype,orientation): # repeat of fitness function but saving theta
    nn = fnn.FNN(layers)
    nn.setParams(genotype)
    body = Vehicle.Vehicle(orientation)
    xpos = np.zeros((len(time)))
    ypos = np.zeros((len(time)))
    body.leftsensor = 0              
    body.rightsensor = 0         
    k=0
    for t in time:
        inp = body.state(lights)
        out = nn.forward(inp)*2-1 + np.random.normal(0.0,noisestd)
        body.round(lights, out)
        body.sense(lights)
        xpos[k] = body.xpos
        ypos[k] = body.ypos
        k += 1
    return xpos,ypos

xpos0,ypos0 = evaluate(bestind_genotype,0)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*1)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*2)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*3)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*4)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*5)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*6)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*7)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*8)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*9)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*10)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*11)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*12)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*13)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*14)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*15)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*16)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*17)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*18)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*19)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*20)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*21)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*22)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*23)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*24)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*25)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*26)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*27)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*28)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*29)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*30)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*31)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*32)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*33)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*34)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')

xpos0,ypos0 = evaluate(bestind_genotype,(1/36)*35)
plt.plot(xpos0,ypos0)
plt.plot(xpos0[-1],ypos0[-1],'ko')


plt.plot(0,0,'ro')
plt.plot(lights.xpos,lights.ypos,'go')
#plt.ylabel("Several different things!")
#plt.legend(["x","vel","theta","ang. vel"])
plt.title("Agent #"+str(1))
plt.show()

