import EA                 
import fnn                
import Vehicle            
import numpy as np
import matplotlib.pyplot as plt
import sys 

id = "8"

# ANN Params
layers = [2,5,5,2]

# Task Params
duration = 100
stepsize = 0.02    
noisestd = 0.01 

# Time
time = np.arange(0.0,duration,stepsize)

# EA Params
popsize = 10
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
recombProb = 0.5
mutatProb = 0.01
tournaments = 25*popsize

lights = Vehicle.Light()

# Fitness initialization ranges
trials_leftsensor = 2
trials_rightsensor = 2
total_trials = trials_leftsensor*trials_rightsensor

leftsensor_range = np.linspace(0,1, num=trials_leftsensor)
rightsensor_range = np.linspace(0,1, num=trials_rightsensor)


#mkae it have 4 orientations or do random tests for averages
# Fitness function
def fitnessFunction(genotype):
    nn = fnn.FNN(layers)
    nn.setParams(genotype)
    body = [Vehicle.Vehicle(0),Vehicle.Vehicle(0.5), Vehicle.Vehicle(0.25), Vehicle.Vehicle(0.75), Vehicle.Vehicle(0.125),Vehicle.Vehicle(0.375),Vehicle.Vehicle(0.625),Vehicle.Vehicle(0.875)]
    #can send nn to the body, use the nn forward
    for ls in leftsensor_range:
        for rs in rightsensor_range:
            body[0].leftsensor = ls
            body[0].rightsensor = rs
            body[1].leftsensor = ls
            body[1].rightsensor = rs
            body[2].leftsensor = ls
            body[2].rightsensor = rs
            body[3].leftsensor = ls
            body[3].rightsensor = rs
            body[4].leftsensor = ls
            body[4].rightsensor = rs
            body[5].leftsensor = ls
            body[5].rightsensor = rs
            body[6].leftsensor = ls
            body[6].rightsensor = rs
            body[7].leftsensor = ls
            body[7].rightsensor = rs
            t = 0
            while t < duration:
                inp0 = body[0].state(lights)
                out0 = nn.forward(inp0)*2 - 1 + np.random.normal(0.0,noisestd)
                body[0].round(lights, out0)
                body[0].sense(lights)

                inp1 = body[1].state(lights)
                out1 = nn.forward(inp1)*2 - 1 + np.random.normal(0.0,noisestd)
                body[1].round(lights, out1)
                body[1].sense(lights)

                inp2 = body[2].state(lights)
                out2 = nn.forward(inp2)*2 - 1 + np.random.normal(0.0,noisestd)
                body[2].round(lights, out2)
                body[2].sense(lights)

                inp3 = body[3].state(lights)
                out3 = nn.forward(inp3)*2 - 1 + np.random.normal(0.0,noisestd)
                body[3].round(lights, out3)
                body[3].sense(lights)

                inp4 = body[4].state(lights)
                out4 = nn.forward(inp4)*2 - 1 + np.random.normal(0.0,noisestd)
                body[4].round(lights, out4)
                body[4].sense(lights)

                inp5 = body[5].state(lights)
                out5 = nn.forward(inp5)*2 - 1 + np.random.normal(0.0,noisestd)
                body[5].round(lights, out5)
                body[5].sense(lights)

                inp6 = body[6].state(lights)
                out6 = nn.forward(inp6)*2 - 1 + np.random.normal(0.0,noisestd)
                body[6].round(lights, out6)
                body[6].sense(lights)

                inp7 = body[7].state(lights)
                out7 = nn.forward(inp7)*2 - 1 + np.random.normal(0.0,noisestd)
                body[7].round(lights, out7)
                body[7].sense(lights)
                t += stepsize
    return (-body[0].distance(lights) - body[1].distance(lights) - body[2].distance(lights) - body[3].distance(lights))/4

# Evolve and visualize fitness over generations
print(tournaments)
ga = EA.MGA(fitnessFunction, genesize, popsize, recombProb, tournaments, mutatProb)
ga.run()
np.save("evol"+id+".npy",ga.bestfit)
np.save("wevol"+id+".npy",ga.worstfit)
np.save("avevol"+id+".npy",ga.avefit)

# Get best evolved network and show its activity
bestind_num = int(ga.bestind[-1])
bestind_genotype = ga.pop[bestind_num]
np.save("gen"+id+".npy",bestind_genotype)
worstind_num = int(ga.worstind[-1])
worstind_genotype = ga.pop[worstind_num]
np.save("wgen"+id+".npy",worstind_genotype)
