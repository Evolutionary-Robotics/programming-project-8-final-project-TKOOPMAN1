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

def evaluate(genotype): # repeat of fitness function but saving theta
    nn = fnn.FNN(layers)
    nn.setParams(genotype)
    body = Vehicle.Vehicle()
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

xpos,ypos = evaluate(bestind_genotype)
wxpos,wypos = evaluate(worstind_genotype)

plt.plot(xpos,ypos)
#plt.xlabel("Time")
plt.plot(lights.xpos,lights.ypos,'go')
plt.plot(0,0,'ro')
plt.plot(xpos[-1],ypos[-1],'ko')
plt.plot(wxpos,wypos)
plt.plot(wxpos[-1],wypos[-1],'yo')
#plt.ylabel("Several different things!")
#plt.legend(["x","vel","theta","ang. vel"])
plt.title("Agent #"+str(1))
plt.show()

