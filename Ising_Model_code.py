import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm 
from matplotlib import colors 
import random as rand

class IsingModel:
    '''a class to investigate the properties of the ising model'''
    def __init__(self, grid_type = 'square', dim = 10, J=[1], pbcs = True, Jrule = 'nn', temp=1, no_vacancies = 0, h = 0):
        
        self.grid_type = grid_type 
        self.dim = dim
        self.pbcs = pbcs
        self.Jrule = Jrule #List of J values
        self.h = h #External Field
        
        self.neighbours = [] #List of neighbours
        
        #Below takes into account what type of grid it is and makes a list of nearest neighbours
        if Jrule == 'nn':
            if self.grid_type == 'BCC': #If you want a square lattice type square!
                self.BCC()
                self.num = 3
                
            if self.grid_type == 'square': #If you want a square lattice type square!
                self.create_square_grid()
                self.num = 2
                
            if self.grid_type == '1D': #If you want a 1D lattice 
                self.create_1D_grid()
                self.num =1

            if self.grid_type == 'triangular': #If you want a triangular lattice
                self.create_triangular_grid()

            if self.grid_type == '3D': #If you want to be fancy and go 3d
                self.num = 3
                self.create_three_grid()
            
            #I created this self.num parameter as there was a wee bit of trouble in the code later on when working with 1d and 3d
            #It made life simpler to just define this function
            if self.grid_type != '3D' and self.grid_type != '1D' and self.grid_type != 'BCC': 
                self.num = 2      
                
        if Jrule == 'nnn' and self.grid_type == 'square': #next nearest neighbours
            self.extra_neighbours()
            self.num = 2
            
        if Jrule == 'nnn' and self.grid_type == 'BCC': #next nearest neighbours
            self.BCC()
            self.num = 3
                
        #Random Initial Moments are calculated   
        self.init_moments = np.random.choice([-1, 1], self.X.shape) #Initial moments are calculated. They are randomised
        self.J = self.create_Jlist(J) #Creates list of J
        self.current_moments = np.copy(self.init_moments) 
        self.magnetisation = self.current_moments.flatten().sum() #Calculates the magnetisation of the whole lattice
        self.maglist = [] #List for average magnetisation per site (will be used over range of temperatures)
        self.maglist.append(abs(self.magnetisation)/self.num_sites) #Calculates average magnetisation per site
        self.energy_list = []
        print(self.current_moments)
        self.current_energy =  self.grid_energy(self.current_moments) #Calculates the energy of the lattice
        self.energy_list.append(self.current_energy/self.num_sites) #Calculates the average energy per lattice site
        
        self.temp = temp #Temperature at which the Ising Model is investigated
       
    def reset(self):
        '''Resets the lattice, randomly giving each point a spin'''
        self.current_moments = np.random.choice([-1, 1], self.Y.shape)

    def extra_neighbours(self):
        '''Gives each point an additional neareat neighbours four neighbours'''
        x, y = np.linspace(0, self.dim-1, self.dim), np.linspace(0, self.dim-1, self.dim)
        X, Y = np.meshgrid(y, x)
        self.X, self.Y = X.flatten(), Y.flatten()
        self.num_sites = self.dim*self.dim
        
        for i, (x1, x2) in enumerate(zip(self.X, self.Y)):
        
            if self.pbcs == True:
                    templist = [[int((x1 - 1) % self.dim), x2] , 
                                [int((x1 + 1) % self.dim), x2] ,
                                [x1, int((x2 - 1) % self.dim) ], 
                                [x1, int((x2 + 1) % self.dim)], 
                                [int((x1 - 1) % self.dim), int((x2 - 1) % self.dim)] ,
                                [int((x1 + 1) % self.dim), int((x2 + 1) % self.dim)] , 
                                [int((x1 + 1) % self.dim),int((x2 - 1) % self.dim) ], 
                                [int((x1 - 1) % self.dim), int((x2 + 1)%(self.dim))] ]
                    #print(templist)
                    reallist = [int(a[1]*self.dim + a[0]) for a in templist] #indices of nearest neighbour
                    self.neighbours.append(reallist)
                    
    def create_1D_grid(self):
        '''Creates a square grid, which will then be used for the lattice investigated'''
        self.X = np.linspace(0, self.dim-1, self.dim)
        self.Y = np.zeros_like(self.X)
        self.num_sites = self.dim
        
        for i, (x1) in enumerate((self.X)):
            if self.pbcs == True:
                templist = [int((x1 - 1) % self.dim), int((x1 + 1) % self.dim) ]
                self.neighbours.append(templist)
                
    def BCC(self):
        '''creates a three dimensional lattice square lattice, also known as the simple cubic lattice'''
        x, y, z = np.linspace(0, (self.dim)-1, self.dim), np.linspace(0, (self.dim)-1, self.dim), np.linspace(0, (self.dim)-1, self.dim)
        X, Y, Z = np.meshgrid(y, x, z)
        self.X, self.Y, self.Z = X.flatten(), Y.flatten(), Z.flatten()
        self.num_sites = self.dim*self.dim*self.dim
        
        if self.Jrule == 'nn':
        
            for i, (x1, x2, x3) in enumerate(zip(self.X, self.Y, self.Z)):
                if self.pbcs == True:
                    templist = [[x1, x2, int((x3 - 1) % self.dim)] ,  #say atom on x1 plane
                                [x1, x2, int((x3 + 1) % self.dim)] , 
                                [x1, int((x2 + 1) % self.dim), int((x3 - 1) % self.dim)], 
                                [x1, int((x2 + 1) % self.dim), int((x3 + 1) % self.dim)],
                                [int((x1 + 1) % self.dim), x2, int((x3 - 1) % self.dim)] ,  #say atom on x1 plane
                                [int((x1 + 1) % self.dim), x2, int((x3 + 1) % self.dim)] , 
                                [int((x1 + 1) % self.dim), int((x2 + 1) % self.dim), int((x3 - 1) % self.dim)], 
                                [int((x1 + 1) % self.dim), int((x2 + 1) % self.dim), int((x3 + 1) % self.dim)]]

                    reallist = [int(a[0]*self.dim + a[1]*self.dim**2 + a[2]) for a in templist] #indices of nearest neighbour

                    self.neighbours.append(reallist)
        
        if self.Jrule == 'nnn':
        
            for i, (x1, x2, x3) in enumerate(zip(self.X, self.Y, self.Z)):
                if self.pbcs == True:
                    templist = [[x1, x2, int((x3 - 1) % self.dim)] ,  #say atom on x1 plane
                                [x1, x2, int((x3 + 1) % self.dim)] , 
                                [x1, int((x2 + 1) % self.dim), int((x3 - 1) % self.dim)], 
                                [x1, int((x2 + 1) % self.dim), int((x3 + 1) % self.dim)],
                                [int((x1 + 1) % self.dim), x2, int((x3 - 1) % self.dim)] ,  #say atom on x1 plane
                                [int((x1 + 1) % self.dim), x2, int((x3 + 1) % self.dim)] , 
                                [int((x1 + 1) % self.dim), int((x2 + 1) % self.dim), int((x3 - 1) % self.dim)], 
                                [int((x1 + 1) % self.dim), int((x2 + 1) % self.dim), int((x3 + 1) % self.dim)], 
                                [x1, int((x2 + 1) % self.dim), x3], [x1, int((x2 - 1) % self.dim), x3], 
                                [int((x1 + 1) % self.dim), x2, x3], [int((x1 - 1) % self.dim), x2, x3]] 
                    #last four is next nearest neighbours

                    reallist = [int(a[0]*self.dim + a[1]*self.dim**2 + a[2]) for a in templist] #indices of nearest neighbour

                    self.neighbours.append(reallist)
        
    def create_square_grid(self):
        '''Creates a square grid, which will then be used for the lattice investigated. Non periodic boundary
        conditions can also be applied to the square grid lattice in this model'''
        x, y = np.linspace(0, self.dim-1, self.dim), np.linspace(0, self.dim-1, self.dim)
        X, Y = np.meshgrid(x, y)
        self.X, self.Y = X.flatten(), Y.flatten()
        self.num_sites = self.dim*self.dim
        
        for i, (x1, x2) in enumerate(zip(self.X, self.Y)):
            
            if self.pbcs == True:
                templist = [[int((x1 - 1) % self.dim), x2] , [int((x1 + 1) % self.dim), x2] , [x1,int((x2 - 1) % self.dim) ], 
                            [x1, int((x2 + 1) % self.dim)] ]
                reallist = [int(a[1]*self.dim + a[0]) for a in templist] 
                self.neighbours.append(reallist)
        
            elif self.pbcs == False: #This is when non periodic conditions are applied. Note that the number of neighbours is not equal
                templist = [[x1 - 1, x2] , [x1+1, x2] , [x1,x2-1], 
                            [x1, x2+1] ] #list of the nearest neighbours
                ar = []
                for a in templist:
                    
                    if a[0] >= 0 and a[1] >= 0 and a[0] < self.dim and a[1] < self.dim:
                        ar.append([a[0], a[1]])
                reallist = [int(c[1]*self.dim + c[0]) for c in ar]
                        
                self.neighbours.append(reallist)
                
            elif self.pbcs == 'Cylindrical':
                templist = [[int((x1 - 1) % self.dim), x2] , [int((x1 + 1) % self.dim), x2] , [x1,x2-1], 
                            [x1, x2+1] ] #list of the nearest neighbours
                ar = []
                for a in templist:
                    if a[1] >= 0 and a[1] < self.dim:
                        ar.append([a[0], a[1]])
                reallist = [int(c[1]*self.dim + c[0]) for c in ar]
                        
                self.neighbours.append(reallist)
                
    def create_triangular_grid(self):
        '''creates a triangular lattice, where each point has six nearest neighbours in the lattice'''
        x, y = np.linspace(0, self.dim-1, self.dim), np.linspace(0, self.dim-1, self.dim)
        X, Y = np.meshgrid(y, x)
        self.X, self.Y = X.flatten(), Y.flatten()
        self.num_sites = self.dim*self.dim
        
        for i, (x1, x2) in enumerate(zip(self.X, self.Y)):
            if self.pbcs == True:
                templist = [[int((x1 - 1) % self.dim), x2] , [int((x1 + 1) % self.dim), x2] , [x1,int((x2 - 1) % self.dim) ], 
                            [x1, int((x2 + 1) % self.dim)],[int((x1 + 1) % self.dim),int((x2 - 1) % self.dim) ], 
                            [int((x1 - 1) % self.dim), int((x2 + 1) % self.dim)] ]
                reallist = [int(a[1]*self.dim + a[0]) for a in templist] #indices of nearest neighbour
                
                self.neighbours.append(reallist)
                
    def create_three_grid(self):
        '''creates a three dimensional lattice square lattice, also known as the simple cubic lattice'''
        x, y, z = np.linspace(0, (self.dim)-1, self.dim), np.linspace(0, (self.dim)-1, self.dim), np.linspace(0, (self.dim)-1, self.dim)
        X, Y, Z = np.meshgrid(y, x, z)
        self.X, self.Y, self.Z = X.flatten(), Y.flatten(), Z.flatten()
        self.num_sites = self.dim*self.dim*self.dim
        
        for i, (x1, x2, x3) in enumerate(zip(self.X, self.Y, self.Z)):
            if self.pbcs == True:
                templist = [[x1, x2, int((x3 - 1) % self.dim)] , 
                            [x1, x2, int((x3 + 1) % self.dim)] , 
                            [int((x1 + 1) % self.dim),x2 , x3], 
                            [int((x1 - 1) % self.dim),x2 , x3],
                           [x1, int((x2 - 1) % self.dim),  x3] , 
                            [x1, int((x2 + 1) % self.dim), x3
                            ] ]
                
                reallist = [int(a[0]*self.dim + a[1]*self.dim**2 + a[2]) for a in templist] #indices of nearest neighbour
                
                self.neighbours.append(reallist)
                
                
    def grid_energy(self, moments): 
        energy = 0
        '''Returns scalar for total energy for a given lattice configuration'''
        momentlist = self.current_moments 
        for i in range((self.dim**self.num)): #Notice here how I am using self.num. When summing over the nieghbours for npbc it was helpful
            #print(self.J)
            energy += 0.5*momentlist[i]*np.sum(self.J[i]*momentlist[self.neighbours[i]]) 
            
            +self.h*np.sum(momentlist) #Note here that an external magnetic field can be applied
        return energy
        
    def create_Jlist(self, J):
        '''This function creates a list of the J value to be used when calculating the energy at each point. Notes that 
        if there is more than one J value, this can only be done when the dimension of the lattice is even'''
        
        if len(J) ==1 and self.Jrule == 'nn': #if there is only one interaction energy
            return np.array([J[0] for i in range((self.dim**self.num))])
        
        if self.Jrule == 'nnn':
            if self.grid_type == 'square':
                '''First four will be 1, next four will be sqrt2 apart'''
                jlist = []
                #First create place for each point
                for i in range(self.dim*self.dim*self.dim): #each point
                    for j in range(4):
                        jlist.append(J[0]/np.sqrt(2))
                    for j in range(4):
                        jlist.append(J[0]/1)
                return jlist
            if self.grid_type == 'BCC':
                jlist = []
                #First create place for each point
                for i in range(self.dim*self.dim*self.dim): #each point
                    for j in range(8):
                        jlist.append(J[0]*np.sqrt(3)/2)
                    for j in range(4):
                        jlist.append(J[0]/1)
                return jlist
                
        #Please note: although this is not necessary, I still created this as I would have liked to look into how different interaction energies between points
        #can effect the curie temperature. Unfortunately I did not have enough time, but I would like to carry this out in the furture and this 
        #J list would then be important as it will tell me the interaction energy between each neighbour
       
    def update_single(self, temp):
        '''Update by going through every point in the lattice'''
        self.current_energy =  self.grid_energy(self.current_moments) #Gets current energy
        newE = self.current_energy #This newE will be updated throughout the loop
        if self.Jrule =='nn' or self.Jrule =='nnn': #Checks what the Jrule is
            for q in range(self.dim**self.num):
                neigh = self.current_moments[self.neighbours[q]]
                neighsum = np.sum(self.J[q]*self.current_moments[q]*neigh)
                deltaE = 2*neighsum + 2*self.h*self.current_moments[q] #energy for a given lattice site
                deltaM = 0 #change in magnetisation 
                #print(deltaE)
                if (deltaE <= 0):
                    newE += deltaE #update energy
                    self.current_moments[q] *= -1 #Flip
                    deltaM = 2*self.current_moments[q]
                else:
                    newE +=0
                    self.current_moments[q] = self.current_moments[q]
                    deltaM = 0

                if(deltaE>0): #Now it is defined according to boltzmann probability
                    random_num = float(rand.uniform(0, 1))
                    if (random_num < float(np.exp(-1*float(deltaE)/float(self.temp)))):
                        newE += deltaE #Update energy
                        self.current_moments[q] *= -1 #flip
                        deltaM = 2*self.current_moments[q] #change in magnetisation
                    else:
                        newE +=0
                        self.current_moments[q] = self.current_moments[q]
                        deltaM = 0

                self.magnetisation += deltaM #Update Magnetisation

        self.current_energy = newE #update energy

        self.energy_list.append(self.current_energy/self.num_sites) #average energy per site
        self.maglist.append(abs(self.magnetisation)/self.num_sites) #Absolute value used due to symmetry of hamiltonian
        
    def run(self, num_its=100, temp=-1): 
        '''Performs multiple updates for a given temperature'''
        self.temp = temp
        for it in range(num_its):
            self.update_single(temp = self.temp)
            
    def grid_figure(self):
        '''This prints the lattice for a given configuration'''
        fig = plt.figure(figsize=(6,5))
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height]) 

        start, stop, n_values = 0, self.dim, self.dim

        x_vals = np.linspace(start, stop, n_values)
        y_vals = np.linspace(start, stop, n_values)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        lattice = np.zeros((self.dim, self.dim))
        n = 0
        for i in range(self.dim):
            for j in range(self.dim):
                lattice[i, j] = self.current_moments[n]
                n +=1

        cp = plt.pcolormesh(X, Y, lattice)

        ax.set_title('100 x 100 Grid')
        plt.show()
        
    def print_equil(self, num_its= 300, temp = -1, to_print = 'all'):
        '''This runs the Ising Model and puts it into equilibrium and prints the average energy and magnetisation list'''
        self.run(temp = temp, num_its = num_its)
        
        if to_print == 'all' or to_print == 'energy':
            plt.plot(self.energy_list)
            plt.grid(linestyle= '-', linewidth=-.2)
            plt.ylabel('Number of Iterations')
            plt.xlabel('Average Energy')
            plt.show()
        
        if to_print == 'all' or to_print == 'magnetisation':
            plt.plot(self.maglist)
            plt.grid(linestyle= '-', linewidth=-.2)
            plt.ylabel('Number of Iterations')
            plt.xlabel('Average Energy')
            plt.show()
        
    def props_with_std(self, tempmin = 0.1, tempmax = 4, to_print = 'all'):
        '''This investigates the various properties of the Ising Model over a wide range of temperatures and prints the properties
        over a wide range of temperatures'''
        testruns, countruns = 100, 100
        self.run(temp = self.temp, num_its =3000)
        templist = np.linspace(tempmin,tempmax, 20) #Input range of temperatures here
        aveng, avmag, stdeng, stdmag = np.zeros(len(templist)), np.zeros(len(templist)), np.zeros(len(templist)), np.zeros(len(templist))
        heat_capacity, mag_suscept = np.zeros(20), np.zeros(20)
        list1, list2, list3, list4 = np.zeros((20, 20)), np.zeros((20, 20)), np.zeros((20, 20)), np.zeros((20, 20))
        for i, temp0 in enumerate(templist):
            for j in range(len(list1)):
                print(j, i)
                self.run(num_its = testruns + countruns, temp=temp0)
                aveng[i] = (np.mean(self.energy_list[-countruns:]))
                avmag[i] = (np.mean(self.maglist[-countruns:]))
                stdeng[i] =  (np.std(self.energy_list[-countruns:]))
                stdmag[i] = (np.std(self.maglist[-countruns:]))
    
                heat_capacity[i] = (stdeng[i]/(temp0**2))
                mag_suscept[i] = (stdmag[i]/(temp0))
                list1[i][j] = np.mean(self.energy_list[-countruns:]) #this repeats it many times and gets the list
                list2[i][j] = np.mean(self.maglist[-countruns:])
                list3[i][j] = heat_capacity[i] #Needed for magnetic susceptibility and heat capacity
                list4[i][j] = mag_suscept[i]
        
        stdE, stdM, stdH, stdS = [], [], [], []
        print(list1[0])
        for i in range(len(list1)):
            for j in range(len(list1)):
        
                if abs(list1[i][j] - list1[i][int((j - 1) % 20)]) > 0.2:
                    list1[i][j] = 0
                
        #now we have to get the standard deviation. Go through array and get each average for given ttemp
        #Now we have to create a list of each of the average of the averages
        for j in range(len(list1)):
            stdE.append(np.std(list1[j]))
            stdM.append(np.std(list2[j]))
            stdH.append(np.std(list3[j]))
            stdS.append(np.std(list4[j]))

        if to_print == 'all' or to_print =='magnetisation':
            plt.errorbar(templist[0:], avmag[0:], yerr = stdM,  marker='o', mfc='red', mec='green')
            plt.grid(linestyle= '-', linewidth=-.2)
    
            plt.xlabel('Temperature J/k')
            plt.show()

        if to_print == 'all' or to_print =='energy':
            #plt.plot(templist[0:], aveng[0:], 'ro')
            plt.errorbar(templist[0:], aveng[0:], yerr = stdE,  marker='o', mfc='red', mec='green')
            plt.grid(linestyle= '-', linewidth=-.2)
            plt.ylabel('Average Energy per Spin')
            plt.xlabel('Temperature J/k')
            plt.show()
            
        if to_print == 'all' or to_print =='heat capacity':
            plt.errorbar(templist[0:], heat_capacity[0:], yerr = stdH,  marker='o', mfc='red', mec='green')
            #plt.plot(templist[0:], heat_capacity[0:], 'ro')
            plt.grid(linestyle= '-', linewidth=-.2)
            plt.ylabel('Heat Capactity')
            plt.xlabel('Temperature J/k')
            plt.show()
            
        if to_print == 'all' or to_print =='magnetic susceptibility':
            #plt.plot(templist[0:], mag_suscept[0:], 'ro')
            plt.errorbar(templist[0:], mag_suscept[0:], yerr = stdS,  marker='o', mfc='red', mec='green', capthick = 2)
            plt.grid(linestyle= '-', linewidth=-.2)
            plt.ylabel('Magnetic Susceptibility')
            plt.xlabel('Temperature J/k')
            
            plt.show()
            
        return aveng, avmag,heat_capacity, mag_suscept
            
    def props(self, tempmin = 0.1, tempmax = 4, to_print = 'all'):
        '''This investigates the various properties of the Ising Model over a wide range of temperatures'''
        testruns, countruns = 100, 200
        templist = np.linspace(tempmin,tempmax, 20) #Input range of temperatures here
        aveng, avmag, stdeng, stdmag = np.zeros(len(templist)), np.zeros(len(templist)), np.zeros(len(templist)), np.zeros(len(templist))
        self.run(temp = self.temp, num_its =100)
        for i, temp0 in enumerate(templist):
            #self.reset()
            print(i)
            self.run(num_its = testruns + countruns, temp=temp0)
            aveng[i] = (np.mean(self.energy_list[-countruns:]))
            avmag[i] = (np.mean(self.maglist[-countruns:]))
            stdeng[i] =  (np.std(self.energy_list[-countruns:]))
            stdmag[i] = (np.std(self.maglist[-countruns:]))

        heat_capacity = stdeng/(templist**2)
        mag_suscept = stdmag/(templist)
        
        if to_print == 'all' or to_print =='energy':
            plt.plot(templist[0:], avmag[0:], 'ro')
            plt.grid(linestyle= '-', linewidth=-.2)
            plt.ylabel('Average Magnetisation per Spin')
            plt.xlabel('Temperature J/k')
            plt.show()

        if to_print == 'all' or to_print =='magnetisation':
            plt.plot(templist[0:], aveng[0:], 'ro')
            plt.grid(linestyle= '-', linewidth=-.2)
            plt.ylabel('Average Energy per Spin')
            plt.xlabel('Temperature J/k')
            plt.show()
            
        if to_print == 'all' or to_print =='heat capacity':
            plt.plot(templist[0:], heat_capacity[0:], 'ro')
            plt.grid(linestyle= '-', linewidth=-.2)
            plt.ylabel('Heat Capactity')
            plt.xlabel('Temperature J/k')
            plt.show()
            
        if to_print == 'all' or to_print =='magnetic susceptibility':
            plt.plot(templist[0:], mag_suscept[0:], 'ro')
            plt.grid(linestyle= '-', linewidth=-.2)
            plt.ylabel('Magnetic Susceptibility')
            plt.xlabel('Temperature J/k')
            plt.show()
            
        return aveng, avmag,heat_capacity, mag_suscept
    
#Equilibrium over various different temperatures
'''
ising1 = IsingModel(dim = 25, pbcs = True, temp = 1, J = [1], h = 0)
ising1.run(temp = 1, num_its = 100)

ising2 = IsingModel(dim = 25, pbcs = True, temp = 2, J = [1], h = 0)
ising2.run(temp = 2, num_its = 100)

ising3 = IsingModel(dim = 25, pbcs = True, temp = 10, J = [1], h = 0)
ising3.run( temp = 10, num_its = 100)

plt.plot(ising1.energy_list, label = 'T = 1')
plt.plot(ising2.energy_list, label = 'T = 2')   
plt.plot(ising3.energy_list, label = 'T = 10')
plt.legend()
plt.title('Energy of lattice vs Number of Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('Average Energy per Site')
plt.show()
'''

#Investigate various properties of the Ising Model
'''
ising1 = IsingModel(dim = 10, pbcs = True, temp = 1, J = [1], h = 0)
ising1.print_equil(temp = 1, num_its = 100)

ising1.props_with_std()
'''

#Ising Model for different dimensions
'''
ising1 = IsingModel(dim = 10, pbcs = True, temp = 1, J = [1], h = 0, grid_type = '1D')
ising1.print_equil(temp = 1, num_its = 50)

ising1.props_with_std(tempmax = 6)

ising2 = IsingModel(dim = 10, pbcs = True, temp = 1, J = [1], h = 0, grid_type = 'square')
ising2.print_equil(temp = 1, num_its = 50)

ising2.props_with_std(tempmax = 6)

ising3 = IsingModel(dim = 10, pbcs = True, temp = 1, J = [1], h = 0, grid_type = '3D')
ising3.print_equil(temp = 1, num_its = 50)

ising3.props_with_std(tempmax = 6)
'''


#Include if you want to see how the size effects the Curie Temperature        

ising_3 = IsingModel(dim = 3, pbcs = True, temp = 1, J = [1], h = 0)
m13 = ising_3.props(tempmax = 10)[1]

m3 = ising_3.maglist[0:]

ising_10 = IsingModel(dim = 10, pbcs = True, temp = 1, J = [1], h = 0)
m110 = ising_10.props(tempmax = 10)[1]

m10 = ising_10.maglist[0:]

ising_25 = IsingModel(dim = 25, pbcs = True, temp = 1, J = [1], h = 0)
m125 = ising_25.props(tempmax = 10)[1]

m25 = ising_25.maglist[0:]

templist = np.linspace(0.1, 10, 20)

plt.plot(templist[0:], m13[0:], 'ro', label = 'N = 3')
plt.plot(templist[0:], m110[0:], 'bo', label = 'N = 10')
plt.plot(templist[0:], m125[0:], 'go', label = 'N = 25')
plt.legend()
plt.xlabel('Temperature')
plt.ylabel('Magnetisation ')
plt.show()


#Include if you want to see how boundary conditions effect the Curie Temp
'''
ising_3 = IsingModel(dim = 10, pbcs = True, temp = 0.1, J = [1], h = 0)
m13 = ising_3.props(tempmax = 4)[1]

m3 = ising_3.maglist[0:]

ising_10 = IsingModel(dim = 10, pbcs = False, temp = 0.1, J = [1], h = 0)
m110 = ising_10.props(tempmax = 4)[1]

m10 = ising_10.maglist[0:]

ising_25 = IsingModel(dim = 10, pbcs = 'Cylindrical', temp = 0.1, J = [1], h = 0)
m125 = ising_25.props(tempmax = 4)[1]

m25 = ising_25.maglist[0:]

templist = np.linspace(0.1, 4, 20)

plt.plot(templist[0:], m13[0:], 'ro', label = 'Perioidic')
plt.plot(templist[0:], m110[0:], 'bo', label = 'Non-Periodic')
plt.plot(templist[0:], m125[0:], 'go', label = 'Cylindrical')
plt.legend()
plt.xlabel('Temperature')
plt.ylabel('Magnetisation ')
plt.show()
'''
#Include this is you want to determine the size of the lattice for pbc and nbc
'''
ising_3 = IsingModel(dim = 10, pbcs = True, temp = 1, J = [1], h = 0)
m13 = ising_3.props(tempmax = 4)[1]

m3 = ising_3.maglist[0:]

ising_10 = IsingModel(dim = 10, pbcs = False, temp = 1, J = [1], h = 0)
m110 = ising_10.props(tempmax = 4)[1]

m10 = ising_10.maglist[0:]

ising_25 = IsingModel(dim = 50, pbcs = False, temp = 1, J = [1], h = 0)
m125 = ising_25.props(tempmax = 4)[1]

m25 = ising_25.maglist[0:]

templist = np.linspace(0.1, 10, 20)

plt.plot(templist[0:], m13[0:], 'ro', label = 'PBC; N = 10')
plt.plot(templist[0:], m110[0:], 'bo', label = 'NPBC; N = 10')
plt.plot(templist[0:], m125[0:], 'go', label = 'NPBC; N = 50')
plt.legend()
plt.xlabel('Temperature J/k')
plt.ylabel('Magnetisation')
plt.show()
'''
'''
#Now we introduce how distsnce dependancy effects the Curie Temperature
#Expect fro larger distances the CT will decrease

ising_3 = IsingModel(dim = 10, pbcs = True, temp = 0.1, J = [1], h = 0)
m13 = ising_3.props(tempmax = 20)[1]

m3 = ising_3.maglist[0:]

ising_10 = IsingModel(dim = 10, pbcs = True, temp = 0.1, J = [2.5], h = 0)
m110 = ising_10.props(tempmax = 20)[1]

m10 = ising_10.maglist[0:]

ising_25 = IsingModel(dim = 10, pbcs = True, temp = 0.1, J = [5], h = 0)
m125 = ising_25.props(tempmax = 20)[1]

m25 = ising_25.maglist[0:]

templist = np.linspace(0.1, 10, 20)

plt.plot(templist[0:], m13[0:], 'ro', label = 'd = 5')
plt.plot(templist[0:], m110[0:], 'bo', label = 'd = 2')
plt.plot(templist[0:], m125[0:], 'go', label = 'd = 5' )
plt.legend()
plt.xlabel('Temperature')
plt.ylabel('Magnetisation')
plt.show()
'''

#Next we do the the dependence between neighest neighbours. 
'''
ising_3 = IsingModel(dim = 10, pbcs = True, temp = 0.1, J = [1], h = 0)
m13 = ising_3.props(tempmax = 20)[1]

m3 = ising_3.maglist[0:]

ising_10 = IsingModel(dim = 10, pbcs = True, temp = 0.1, J = [1], h = 0, Jrule = 'nnn')
m110 = ising_10.props(tempmax = 20)[1]

plt.plot(templist[0:], m13[0:], 'ro', label = 'First NN')
plt.plot(templist[0:], m110[0:], 'bo', label = 'First and Second NN')
plt.xlabel('Temperature J/k')
plt.ylabel('Magnetisation')
plt.legend()
plt.show()
'''

#Now we investigate BCC vs Square
'''
ising_bcc = IsingModel(dim = 10, pbcs = True, temp = 1, J = [1], h = 0, Jrule = 'nnn', grid_type = 'BCC')
m13 = ising_bcc.props(tempmax = 20)[1]


ising_bcc2 = IsingModel(dim = 10, pbcs = True, temp = 1, J = [1], h = 0, Jrule = 'nn', grid_type = 'BCC')
m132 = ising_bcc2.props(tempmax = 20)[1]


ising_s = IsingModel(dim = 10, pbcs = True, temp = 1, J = [1], grid_type = '3D', Jrule = 'nn')
m110 = ising_s.props(tempmax = 20)[1]

plt.plot(templist[0:], m13[0:], 'ro', label = 'BCC with NNN')
plt.plot(templist[0:], m132[0:], 'go', label = 'BCC')
plt.plot(templist[0:], m110[0:], 'bo', label = 'Cubic')
plt.xlabel('Temperature J/k')
plt.ylabel('Magnetisation')
plt.legend()
plt.show()
'''

#Triangular 2d vs square 2d
'''
ising_3 = IsingModel(dim = 10, pbcs = True, temp = 0.1, J = [1], h = 0, grid_type = 'triangular')
m13 = ising_3.props(tempmax = 20)[1]

m3 = ising_3.maglist[0:]

ising_10 = IsingModel(dim = 10, pbcs = True, temp = 0.1, J = [1], h = 0)
m110 = ising_10.props(tempmax = 20)[1]

plt.plot(templist[0:], m13[0:], 'ro', label = 'Triangular 2D grid')
plt.plot(templist[0:], m110[0:], 'bo', label = 'Square 2D grid')
plt.xlabel('Temperature J/k')
plt.ylabel('Magnetisation')
plt.legend()
plt.show()
'''