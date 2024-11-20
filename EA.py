import numpy as np
import matplotlib.pyplot as plt

class MGA():

    def __init__(self,funct,geneamt,population,probcomb,rounds,probmut):
        self.geneamt = geneamt
        self.population = population
        self.probcomb = probcomb
        self.rounds = rounds
        self.probmut = probmut
        self.funct = funct
        self.pop = np.random.random((population,geneamt))*2 - 1
        # stats for evaluation at end of run time
        self.fits = self.calculateFitness()
        generations = rounds//population
        self.bestfit = np.zeros(generations)
        self.avefit = np.zeros(generations)
        self.worstfit = np.zeros(generations)
        self.bestind = np.zeros(generations)
        self.worstind = np.zeros(generations)
    #this calculates the fitness of all the members of population, saving time later, and making comparison easier
    def calculateFitness(self):
        fits = np.zeros(self.population)
        for i in range(self.population):
            fits[i] = self.funct(self.pop[i])
        return fits

    def run(self):
        generation = 0
        #loop rounds, pick 2 fight, pick winner, do recomb and mut
        for t in range(self.rounds):
            [a,b] = np.random.choice(np.arange(self.population),2,replace=False)
            if self.fits[a] > self.fits[b]: 
                winner = a 
                loser = b
            else:
                winner = b
                loser = a
            for g in range(self.geneamt):
                if np.random.random() < self.probcomb:
                    self.pop[loser][g] = self.pop[winner][g]
                if np.random.random() < self.probmut:
                    self.pop[loser][g] = np.random.random()*2 - 1
            #update fit of the loser
            self.fits[loser] = self.funct(self.pop[loser])
            if t % self.population == 0:
                self.bestfit[generation] = np.max(self.fits)
                self.avefit[generation] = np.average(self.fits)
                self.worstfit[generation] = np.min(self.fits)
                self.bestind[generation] = np.argmax(self.fits)
                self.worstind[generation] = np.argmin(self.fits)
                generation += 1
                print(t,"generation ",np.max(self.fits),"max ",np.mean(self.fits),"mean ",np.min(self.fits),"min ")
    
    def showFitness(self):
        plt.plot(self.bestfit,label="Best")
        plt.plot(self.avefit,label="Average")
        plt.plot(self.worstfit,label="Worst")
        plt.xlabel("Generation Number")
        plt.ylabel("Fitness Value")
        plt.title("Evolution of the Network")
        plt.legend(loc='lower right')
        plt.show()


