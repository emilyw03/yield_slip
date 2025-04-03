# This is currently a prototype of object-oriented python package for study
# multi-electron transfer kinetics
# Some functions might be revised accordingly

# Import packages
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg
from collections import defaultdict as Dict


class Cofactor():
    def __init__(self, name: str, redox: list):
        """
        Initialize this cofactor object, with property: name, and redox potentials
        Arguments:
            name {str} -- Name of the cofactor
            redox {list} -- List of ORDERED redox potential for different redox states
        """
        self.name = name
        self.redox = redox          #(ex.) "[first reduction potential (0 -> 1), second reduction potential (1 -> 2),...]
        self.capacity = len(redox)    # The number of electrons the site can occupy is equal to the number of reduction potentials

    def __str__(self) -> str:         #__str__:a built-in function that computes the "informal" string representations of an object
        """
        Return a string representation of the cofactor
        Returns:
            str -- String representation of the cofactor
        """
        s = ""
        # Initialize with cofactor name
        s += "Cofactor Name: {}\n".format(self.name)     #\n:new line in string
        s += "------------ \n"     #Draw a line between cofactor info (looks cuter!)
        # Print cofactor info, with state_id and relative redox potential
        for i in range(len(self.redox)):
            s += "Redox State ID: {}, Oxidation Potential: {}\n".format(i, self.redox[i])

        return s


class Network():
    def __init__(self):
        """
        Initialize the whole system
        NOTICE: the initialized Network instance has nothing in it, use other functions to insert information
        """
        # system-specific data structure and parameters
        self.num_cofactor = 0
        self.num_state = 1
        self.Elist = []
        self.Midptlist = []
        self.id2cofactor = dict()  # key-value mapping is id-cofactor
        self.cofactor2id = dict()  # key-value mapping is cofactor-id
        self.adjacencyList = list()
        self.D = None  # not defined
        self.K = None  # not defined
        self.siteCapacity = []  # key-value mapping is id-site_capacity
        self.num_reservoir = 0
        self.num_reservoir_mod = 0
        self.reservoirInfo = dict()    # key-value mapping is id-reservoir name, cofactor, redox_state, num_electron, deltaG, rate
        self.reservoirInfoMod = dict()    # key-value mapping is id-reservoir name, cofactor, redox_state, num_electron, deltaG, rate
        self.id2reservoir=dict()    # key-value mapping is id-reservoir name
        self.id2reservoirmod=dict()    # key-value mapping is id-reservoir name
        self.reservoir2id=dict()    # key-value mapping is reservoir name-id
        self.reservoir2idmod=dict()    # key-value mapping is reservoir name-id
        """
        ET-specific data structure and parameters     #Incorporate to the ET function?
        """
        self.hbar = 6.5821 * 10 ** (-16)  # unit: eV sec
        self.beta = 39.06  # unit: 1/kT in 1/eV
        self.reorgE = 0.7
        self.V = 0.1
        # self.reservoir_rate = 10**7 #This is the default rate constant for electron flow INTO the reservoirs. It is currently an input to allow more flexibility
        # self.cofactor_distance = 10 #distance between neighboring cofactors
        # self.slope = 0.1 #this is the "slope" of the energy landscapes (i.e. the difference in reduction potentials of neighboring cofactors)
        # self.N = 100 #The number of points to be plotted
        # self.res2emin = -0.1 #The range of energies of the 2-electron (D) reservoir to be plotted over
        # self.res2emax = 0.1
        # self.dx = (res2emax-res2emin)/N #energy step size


    def __str__(self) -> str:
        """
        Return a string representation of the defined Network
        Returns:
            str -- String representation of the Network
        """
        s = ""
        # 1. print Cofactors information
        s += "Total number of cofactors in the Network: {}\n".format(self.num_cofactor)
        if self.num_cofactor == 0:
            s += "There are no cofactors in the Network, please add cofactors first!\n"
            return s
        for idx, cofactor in self.id2cofactor.items():
            s += "ID: {}\n".format(idx)
            s += cofactor.__str__() 
        # 2. print Adjacency matrix information
        if isinstance(self.D, type(None)):
            s += "------------\n"
            s += "The adjacency matrix has not been calculated yet!\n"
            s += "------------\n"
            return s
        s += "------------\n"
        s += "Adjacency matrix for the Network\n"
        s += "------------\n"
        s += self.D.__str__()
        # 3. print Reservoir information
        if self.num_reservoir == 0:
            s += "------------\n"
            s += "There are no reservoir defined in this system!\n"
            s += "------------\n"
        else:
            s += "------------\n"
            s += "There are {} reservoirs in this system.\n"
            s += "------------\n"
            for res_id, info in self.reservoirInfo.items():
                name, cofactor, redox_state, num_electron, deltaG, rate = info
                s += "------------\n"
                s += "Reservoir ID: {}, Reservoir Name: {}, connects with Cofactor ID {} with Redox State {}\n".format(res_id, name, self.cofactor2id[cofactor], redox_state)
                s += "Number of electron it exchanges at a time: {}\n".format(num_electron)
                s += "Delta G for transfering electron: {}\n".format(deltaG)
                s += "ET rate: {}\n".format(rate)
                s += "------------\n"

        return s

    def addCofactor(self, cofactor: Cofactor):
        """
        Add cofactor into this Network
        Arguments:
            cofactor {Cofactor} -- Cofactor object
        """
        self.num_state *= (cofactor.capacity +1)        # The total number of possible states is equal to the product of sitecapacities+1 of each site.
                                             # (ex.) "Cofactor_1":0,1, "Cofactor_2":0,1,2 -> num_states=(cap_1+1)*(cap_2+1)=(1+1)*(2+1)=2*3=6

        self.id2cofactor[self.num_cofactor] = cofactor   #Starts with self.num_cofactor=0, Gives an ID to cofactors that are added one by one
        self.cofactor2id[cofactor] = self.num_cofactor   #ID of the cofactor just added is basically equal to how many cofactors present in the network
        # self.siteCapacity[self.num_cofactor] = cofactor.capacity
        self.siteCapacity.append(cofactor.capacity)    #Trajectory of cofactor -> id -> capacity of cofactor
        self.num_cofactor += 1    #The number of cofactor counts up

    def addConnection(self, cof1: Cofactor, cof2: Cofactor, distance: float):
        """
        "Physically" connect two cofactors in the network, allow electron to flow
        Arguments:
            cof1 {Cofactor} -- Cofactor instance
            cof2 {Cofactor} -- Cofactor instance
            distance {float} -- Distance between two cofactors, unit in angstrom
        """
        self.adjacencyList.append((self.cofactor2id[cof1], self.cofactor2id[cof2], distance))  #Append ID of cof1, ID of cof2 and distance between cof1 and cof2 to adjacency list 

    def addReservoir(self, name: str, cofactor: Cofactor, redox: int, num_electron: int, deltaG: float, rate: float):
        """
        Add an electron reservoir to the network: which cofactor it exchanges electrons with, how many electrons are exchanged at a time, the deltaG of the exchange, and the rate
        Arguments:
            name {str} -- Name of the reservoir
            cofactor {Cofactor} -- Cofactor the reservoir exchanges electrons
            redox {int} -- Redox state of the cofactor that exchanges electrons with 
            num_electron {int} -- Number of electrons exchanged at a time
            deltaG {float} -- DeltaG of the exchange
            rate {float} -- In rate
        """
        # key: (reservoir_id, cofactor_id)
        # value: list of six variables, [name, cofactor, redox_state, num_electron, deltaG, rate]
        self.id2reservoir[self.num_reservoir] = name
        self.reservoir2id[name] = self.num_reservoir
        self.reservoirInfo[self.num_reservoir] = [name, cofactor, redox, num_electron, deltaG, rate]
        self.num_reservoir += 1

    def addReservoirMod(self, name: str, cofactor: Cofactor, redox: int, num_electron: int, deltaG: float, cofactor1: Cofactor, cofactor2: Cofactor, rate1: float, redox11: int, redox12: int, rate2: float, redox21: int, redox22: int, rate3: float, redox31: int, redox32: int, rate4: float, redox41: int, redox42: int):
        """
        Add an electron reservoir to the network: which cofactor it exchanges electrons with, how many electrons are exchanged at a time, the deltaG of the exchange, and the rate
        Arguments:
            name {str} -- Name of the reservoir
            cofactor {Cofactor} -- Cofactor the reservoir exchanges electrons
            redox {int} -- Redox state of the cofactor that exchanges electrons with 
            num_electron {int} -- Number of electrons exchanged at a time
            deltaG {float} -- DeltaG of the exchange
            rate {float} -- In rate
        """
        # key: (reservoir_id, cofactor_id)
        # value: list of six variables, [name, cofactor, redox_state, num_electron, deltaG, rate]
        self.id2reservoirmod[self.num_reservoir_mod] = name
        self.reservoir2idmod[name] = self.num_reservoir_mod
        self.reservoirInfoMod[self.num_reservoir_mod] = [name, cofactor, redox, num_electron, deltaG, cofactor1, cofactor2, rate1, redox11, redox12, rate2, redox21, redox22, rate3, redox31, redox32, rate4, redox41, redox42]
        self.num_reservoir_mod += 1


    def checkReservoirMod(self):
        modreservoirlist = []
        modreservoirlistrule = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for reservoir_id, infomod in self.reservoirInfoMod.items():
            modreservoirlist.append([infomod[8], infomod[9]])
            modreservoirlist.append([infomod[11], infomod[12]])
            modreservoirlist.append([infomod[14], infomod[15]])
            modreservoirlist.append([infomod[17], infomod[18]])
        if sorted(modreservoirlist) == sorted(modreservoirlistrule):
            pass
        else:
            print('modreservoirlist does not match modreservoirlistrule')
            exit()


    def evolve(self, t: float, pop_init: np.array) -> np.array:
        """
        Evolve the population vector with a timestep of t
        Arguments:
            t {float} -- Time
        Keyword Arguments:
            pop_init {numpy.array} -- Initial population vector (default: None)
        Returns:
            numpy.array -- Final population vector 
        """
        if pop_init is None:
            # if no pop_init is given in the input, give a default initialization
            pop_init = np.zeros(self.num_state)
            # this is the initialization for 1-e case
            pop_init[0] = 1

        return linalg.expm(self.K * t) @ pop_init




    def evolve_new2(self, K_new: np.array, U_new: np.array, U_new_inv: np.array, t: float, pop_init: np.array) -> np.array:
        """
        Evolve the population vector with a timestep of t
        Arguments:
            K_new {mat} -- diagonalized K
            U_new {mat} -- lines up P to match new K
            t {float} -- Time
        Keyword Arguments:
            pop_init {numpy.array} -- Initial population vector (default: None)
        Returns:
            numpy.array -- Final population vector 
        """
        if pop_init is None:
            # if no pop_init is given in the input, give a default initialization
            pop_init = np.zeros(self.num_state)
            # this is the initialization for 1-e case
            pop_init[0] = 1

        return U_new @ linalg.expm(K_new * t) @ U_new_inv @ pop_init

    def evolve_new3(self, K_eig: np.array, U_new: np.array, U_new_inv: np.array, t: float, pop_init: np.array) -> np.array:
        """
        Evolve the population vector with a timestep of t
        Arguments:
            K_eig {vec} -- K eigenvalues
            U_new {mat} -- lines up P to match new K
            t {float} -- Time
        Keyword Arguments:
            pop_init {numpy.array} -- Initial population vector (default: None)
        Returns:
            numpy.array -- Final population vector 
        """
        if pop_init is None:
            # if no pop_init is given in the input, give a default initialization
            pop_init = np.zeros(self.num_state)
            # this is the initialization for 1-e case
            pop_init[0] = 1

        return U_new @ np.diag(np.exp(K_eig * t)) @ U_new_inv @ pop_init



    def ET(self, deltaG: float, R: float, reorgE, beta, V) -> float:
        """
        Calculate the nonadiabatic ET rate according to Marcus theory
        Arguments:
            deltaG {float} -- reaction free energy, unit: eV
            R {float} -- distance for decay factor, unit: angstrom
            reorgE {float} -- reorganization energy, unit: eV
            beta {float} -- inverse of kT, unit: 1/eV
            V {float} -- electronic coupling, unit: eV
        Returns:
            float -- nonadiabatic ET rate, unit: 1/s
        """
        return (2*math.pi/self.hbar)*(self.V**2)*np.exp(-R)*(1/(math.sqrt(4*math.pi*(1/beta)*reorgE)))*np.exp(-beta*(deltaG + reorgE)**2/(4*reorgE))

    def constructAdjacencyMatrix(self):
        """
        Build adjacency matrix from the adjacency list
        """
        # obtain the dimension of this matrix
        dim = self.num_cofactor
        self.D = np.zeros((dim, dim), dtype=float)
        for item in self.adjacencyList:
            id1, id2, distance = item
            # we allow electron to flow back and forth between cofactors, thus D matrix is symmetric
            self.D[id1][id2] = self.D[id2][id1] = distance


    ####################################################
    ####  Core Functions for Building Rate Matrix   ####
    ####################################################
    
    # For the following functions, we make use of the internal labelling of the
    # states which uses one index which maps to the occupation number
    # representation [n1, n2, n3, ..., nN] and convert to idx in the rate
    # back and forth with state2idx() and idx2state() functions.




    def makeMidptlist(self) -> list:
    # use redox list to make midpt list
        self.Midptlist = []
        for n in range(self.num_cofactor):
            cof = self.id2cofactor[n]
            # specific to KT's approach
            # if cof is not a 'reservoir'
            if cof.name != "DR":
                templist = [0.0]
                tempmp = 0.0
                for r in range(cof.capacity):
                    tempmp += cof.redox[r]
                    templist.append(tempmp)
                self.Midptlist.append(templist)
            # if 'cof' is a 'reservoir'
            if cof.name == "DR":
                templist = [0.0] #[cof.redox[0]]
                tempmp = 0.0
                for r in range(cof.capacity):
                    tempmp += cof.redox[0]/2.0
                    templist.append(tempmp)
#                for r in range(cof.capacity):
#                    templist.append(cof.redox[0])
                self.Midptlist.append(templist)
                #print('cof:', cof)
                #print('Midptlist', self.Midptlist[n])
        return(self.Midptlist)

    def makeEnergylist(self) -> list:
    # make energy list for all microstates assuming no interactions
        Elisttemp = 0
        self.Elist = []
        for i in range(self.num_state):
            for n in range(self.num_cofactor):
                cof = self.id2cofactor[n]
                Elisttemp += self.Midptlist[n][self.idx2state(i)[n]]
            self.Elist.append(Elisttemp)
            Elisttemp = 0
        return(self.Elist)


    def makeEnergylist_KT(self) -> list:
    # make energy list for all microstates assuming no interactions.
        self.Elist = []
        for i in range (self.num_state):
            microstate_energy = 0
            for cof_id in range(self.num_cofactor):
                redox_state = self.idx2state(i)[cof_id]
                microstate_energy += self.getCofactorEnergy(cof_id, redox_state)
            self.Elist.append(microstate_energy)
    
        return self.Elist


    def getCofactorEnergy(self, cof_id, redox_state):
    # Calculate the energy of the cofactor when the redox state is "redox_state".
        cof = self.id2cofactor[cof_id]
        energy = 0
        if redox_state == 0:
            energy += 0     # Energy is 0 when the cofactor is fully oxidized
        else:
            for n in range(redox_state):
                energy += -cof.redox[n]    # Energy is \sum_{x+1} (-(x+1th redox potential)) x -> x+1 

        return energy



    def state2idx(self, state: list) -> int:
        """
        Given the list representation of the state, return index number in the main rate matrix
        Arguments:
            state {list} -- List representation of the state
        Returns:
            int -- Index number of the state in the main rate matrix
        """
        idx = 0
        N = 1
        for i in range(self.num_cofactor):
            idx += state[i] * N
            N *= (self.siteCapacity[i] + 1)
        
        return idx

    def idx2state(self, idx: int) -> list:
        """
        Given the index number of the state in the main rate matrix, return the list representation of the state
        Arguments:
            idx {int} -- Index number of the state in the main rate matrix
        Returns:
            list -- List representation of the state
        """
        state = []
        for i in range(self.num_cofactor):
            div = self.siteCapacity[i] + 1
            idx, num = divmod(idx, div)
            state.append(num)

        return state

    def connectStateRate(self, cof_i: int, red_i: int, cof_f: int, red_f: int, k: float, deltaG: float, num_electrons: int):
        """
        Add rate constant k between electron donor (cof_i) a nd acceptor (cof_f) with initial redox state and final redox state stated (red_i, red_f)
        ADDITION: this function combine with detailed balancing feature, helps to save initialization time.
        Arguments:
            cof_i {int} -- Donor cofactor ID
            red_i {int} -- Redox state for donor ID
            cof_f {int} -- Acceptor cofactor ID
            red_f {int} -- Redox state for acceptor ID
            k {float} -- forward state
            deltaG {float} -- deltaG between initial state and final state
        """
        for i in range(self.num_state):
            # loop through all states, to look for initial (donor) state
            if self.idx2state(i)[cof_i] == red_i and self.idx2state(i)[cof_f] == red_f:
                """
                ex. idx:some number -> state:[0 1 1 0 2 3 ...]
                    idx2state(i)[cof_i] -> ith element of the state:[0 1 1 0 2 3...]
                Basically, this "if" statement means: 
                "If cof_ith element of the state:[0 1 1 0 2 3...] is equal to the redox state of the cof_i" and also 
                "If cof_fth element of the state:[0 1 1 0 2 3...] is equal to the redox state of the cof_f"
                """

                for j in range(self.num_state):
                    # loop through all states, to look for final (acceptor) state
                    if self.idx2state(j)[cof_i] == red_i - num_electrons and self.idx2state(j)[cof_f] == red_f + num_electrons:
                        """
                        ex. idx:some number -> state:[0 1 1 0 2 3 ...]
                            idx2state(i)[cof_i] -> ith element of the state:[0 1 1 0 2 3...]
                        Basically, this "if" statement means: 
                        "If cof_ith element of the state:[0 1 1 0 2 3...] is equal to the (redox state - 1) (donates electron so this cofactor is oxidized) of the cof_i" and also 
                        "If cof_fth element of the state:[0 1 1 0 2 3...] is equal to the (redox state + 1) (accepts electron so this cofactor is reduced) of the cof_f"
                           """
                        # initial, final state found! check other electron conservation
                        I = np.delete(self.idx2state(i), [cof_i, cof_f])     # Do not allow changes other than cof_i and cof_f we have searched for
                        J = np.delete(self.idx2state(j), [cof_i, cof_f])     # Deleting the cof_i and cof_f that is already checked to be correct
                        if np.array_equal(I, J):   # Checking that sites other than cof_i and cof_f hasn't changed
                            # i and j state found!
                            kf = k  # forward rate
                            kb = k * np.exp(self.beta*deltaG)
                            self.K[j][i] += kf  # add population of final state, forward process
                            self.K[i][i] -= kf  # remove population of initial state, forward process   #Diagonal elements are the negative sum of the other elements in the same column
                            self.K[i][j] += kb  # add population of initial state, backward process
                            self.K[j][j] -= kb  # remove population of final sate, backward process

    def connectReservoirRate(self, cof_id: int, red_i: int, red_f: int, k: float, deltaG: float):
        """
        Add rate constant k between red_i and red_f of a cofactor, which is connected to a reservoir
        ADDITION: this function combine with detailed balancing feature, helps to save initialization time.
        Arguments:
            cof_id {int} -- Cofactor ID
            red_i {int} -- Redox state for cofactor
            red_f {int} -- Redox state for cofactor
            k {float} -- forward state
            deltaG {float} -- deltaG between initial state and final state
        """
        for i in range(self.num_state):
            # loop through all states, to look for initial (donor) state
            if self.idx2state(i)[cof_id] == red_i:
                """
                ex. idx:some number -> state:[0 1 1 0 2 3 ...]
                idx2state(i)[cof_i] -> ith element of the state:[0 1 1 0 2 3...]
                Basically, this "if" statement means:
                "If cof th element of the state:[0 1 1 0 2 3...] is equal to the redox state of the cofactor"
                """
                for j in range(self.num_state):
                    # loop through all states, to look for final (acceptor) state
                    if self.idx2state(j)[cof_id] == red_f:
                        """
                        ex. idx:some number -> state:[0 1 1 0 2 3 ...]
                        idx2state(i)[cof_i] -> ith element of the state:[0 1 1 0 2 3...]
                        Basically, this "if" statement means: 
                        "If cof th element of the state:[0 1 1 0 2 3...] is equal to the redox state of the final cofactor"
                        """
                        # initial, final state found! check other electron conservation
                        I = np.delete(self.idx2state(i), [cof_id])
                        J = np.delete(self.idx2state(j), [cof_id])
                        if np.array_equal(I, J):
                            # i and j state found!
                            kf = k  # forward rate
                            kb = k * np.exp(self.beta*deltaG)
                            self.K[j][i] += kf  # add population of final state, forward process
                            self.K[i][i] -= kf  # remove population of initial state, forward process
                            self.K[i][j] += kb  # add population of initial state, backward process
                            self.K[j][j] -= kb  # remove population of final state, backward process



    def connectReservoirRateMod(self, cof_id: int, red_i: int, red_f: int, deltaG: float, cof1: int, cof2: int,
                                                                           k1: float, charge11: int, charge12: int,
                                                                           k2: float, charge21: int, charge22: int,
                                                                           k3: float, charge31: int, charge32: int,
                                                                           k4: float, charge41: int, charge42: int):
        """
        Add rate constant k between red_i and red_f of a cofactor, which is connected to a reservoir
        ADDITION: this function combine with detailed balancing feature, helps to save initialization time.
        Arguments:
            cof_id {int} -- Cofactor ID
            red_i {int} -- Redox state for cofactor
            red_f {int} -- Redox state for cofactor
            deltaG {float} -- deltaG between initial state and final state
            cof1 {int} -- Cofactor ID1 that influences rate
            cof2 {int} -- Cofactor ID2 that influences rate
            k1 {float} -- forward state
            charge11 {int} -- charge on cof11
            charge12 {int} -- charge on cof12
            k2 {float} -- forward state
            charge21 {int} -- charge on cof21
            charge22 {int} -- charge on cof22
            k3 {float} -- forward state
            charge31 {int} -- charge on cof31
            charge32 {int} -- charge on cof32
            k4 {float} -- forward state
            charge41 {int} -- charge on cof41
            charge42 {int} -- charge on cof42
        """
        for i in range(self.num_state):
            # loop through all states, to look for initial (donor) state
            if self.idx2state(i)[cof_id] == red_i:
                """
                ex. idx:some number -> state:[0 1 1 0 2 3 ...]
                idx2state(i)[cof_i] -> ith element of the state:[0 1 1 0 2 3...]
                Basically, this "if" statement means:
                "If cof th element of the state:[0 1 1 0 2 3...] is equal to the redox state of the cofactor"
                """
                for j in range(self.num_state):
                    # loop through all states, to look for final (acceptor) state
                    if self.idx2state(j)[cof_id] == red_f:
                        """
                        ex. idx:some number -> state:[0 1 1 0 2 3 ...]
                        idx2state(i)[cof_i] -> ith element of the state:[0 1 1 0 2 3...]
                        Basically, this "if" statement means: 
                        "If cof th element of the state:[0 1 1 0 2 3...] is equal to the redox state of the final cofactor"
                        """
                        # initial, final state found! check other electron conservation
                        I = np.delete(self.idx2state(i), [cof_id])
                        J = np.delete(self.idx2state(j), [cof_id])
                        if np.array_equal(I, J):
                            # i and j state found!
                            if self.idx2state(i)[cof1] == charge11 and self.idx2state(i)[cof2] == charge12:
                                k = k1
                            elif self.idx2state(i)[cof1] == charge21 and self.idx2state(i)[cof2] == charge22:
                                k = k2
                            elif self.idx2state(i)[cof1] == charge31 and self.idx2state(i)[cof2] == charge32:
                                k = k3
                            elif self.idx2state(i)[cof1] == charge41 and self.idx2state(i)[cof2] == charge42:
                                k = k4
                                # charges match microstate of interest
                            kf = k  # forward rate
                            kb = k * np.exp(self.beta*deltaG)
                            self.K[j][i] += kf  # add population of final state, forward process
                            self.K[i][i] -= kf  # remove population of initial state, forward process
                            self.K[i][j] += kb  # add population of initial state, backward process
                            self.K[j][j] -= kb  # remove population of final state, backward process



    def addMultiElectronConnection(self, cof_i, cof_f, donor_state: int, acceptor_state: int, num_electrons, k):
        i = self.cofactor2id[cof_i]
        f = self.cofactor2id[cof_f]   # Finding the name of cofactor of the ijth of the adjacency matrix
        deltaG = sum([cof_i.redox[donor_state-num_electrons + n] - cof_f.redox[acceptor_state+n] for n in range(0, num_electrons)])
        self.connectStateRate(i, donor_state, f, acceptor_state, k, deltaG, num_electrons)   #Adding the rate constant to rate matrix

    def constructRateMatrix(self):
        """
        Build rate matrix (This need to be implemented further)
        """
        # initialize the rate matrix with proper dimension
        self.K = np.zeros((self.num_state, self.num_state), dtype=float)      #The dimension of the rate matrix is basically equal to the total number of states
        # loop through cofactor_id in adjacency matrix
        """
        Take the adjacency matrix which is weighted by the distance to construct the full rate matrix
        """
        for i in range(self.num_cofactor):
            for j in range(i+1, self.num_cofactor):   # These two "for" loops take care of (upper triangular - diagonal) part of the adjacency matrix
                if self.D[i][j] != 0:  # cofactor i and j are connected!  !=:not equal to
                    cof_i = self.id2cofactor[i]
                    cof_f = self.id2cofactor[j]   # Finding the name of cofactor of the ijth of the adjacency matrix
                    dis = self.D[i][j]   #Distance between cof_i and cof_f is the ij th element of the adjacency matrix
                    """
                    Looping through all the possible transfers from donor to acceptor to find their reduction potentials to get deltaG of that transfer. 
                    You use that deltaG to get the Marcus rate of that transfer, and then add that rate constant to the rate matrix.
                    """
                    for donor_state in range(1, cof_i.capacity+1):    #This is correct!!!! Python is weird      #cof.capacity=maximum number of electrons the cofactor can occupy
                        for acceptor_state in range(0, cof_f.capacity):    #This is correct!!!! Python is weird
                            deltaG = cof_i.redox[donor_state-1] - cof_f.redox[acceptor_state]   #This is correct!!!! Python is weird
                            k = self.ET(deltaG, dis, self.reorgE, self.beta, self.V)
                            self.connectStateRate(i, donor_state, j, acceptor_state, k, deltaG,1)   #Adding the rate constant to rate matrix. The last parameter is 1 because these are all 1-electron transfers!
        # loop through reservoirInfo to add reservoir-related rate
        for reservoir_id, info in self.reservoirInfo.items():
            #name, cofactor, redox_state, num_electron, deltaG, rate = info
            #cofactor_dict1 = cofactor
            #cofactor = info[1]
            cofactor_dict1 = info[1]
            for reservoir_id_mod, infomod in self.reservoirInfoMod.items():
#                name, cofactor, redox_state, num_electron, deltaG, cofactor1, cofactor2,
#                ratemod1, redox11, redox12,
#                ratemod2, redox21, redox22,
#                ratemod3, redox31, redox32,
#                ratemod4, redox41, redox42 = infomod
                cofactor_dict2 = infomod[1]
                if cofactor_dict1 == cofactor_dict2:
                    print('reservoir and modreservoir use the same cofactor!  Rates will get overwritten.')
                    exit()
        for reservoir_id, info in self.reservoirInfo.items():
            name, cofactor, redox_state, num_electron, deltaG, rate = info
            cof_id = self.cofactor2id[cofactor]
            final_redox_state = redox_state - num_electron
            self.connectReservoirRate(cof_id, redox_state, final_redox_state, rate, deltaG)
        for reservoir_id, infomod in self.reservoirInfoMod.items():
            #print(infomod[5])
            name, cofactor, redox_state, num_electron, deltaG, cofactor1, cofactor2, rate1, redox11, redox12, rate2, redox21, redox22, rate3, redox31, redox32, rate4, redox41, redox42=infomod
            cof_id = self.cofactor2id[cofactor]
            final_redox_state = redox_state - num_electron
            cof1 = self.cofactor2id[cofactor1]
            cof2 = self.cofactor2id[cofactor2]
            self.connectReservoirRateMod(cof_id, redox_state, final_redox_state, deltaG, cof1, cof2,
                                                                                 rate1, redox11, redox12,
                                                                                 rate2, redox21, redox22,
                                                                                 rate3, redox31, redox32,
                                                                                 rate4, redox41, redox42)

    ########################################
    ####    Data Analysis Functions     ####
    ########################################

    def population(self, pop: np.array, cofactor: Cofactor, redox: int) -> float:
        """
        Calculate the population of a cofactor in a given redox state
         -> (ex.)pop=[1 0 0 2 5 ...]:len(pop)=num_state, pop is the population vector of the states. (pop[0]=population of state[0], pop[1]=population of state[1]...)

        Arguments:
            pop {numpy.array} -- Population vector     This is the population vector of the states. len(pop)=self.num_state
            cofactor {Cofactor} -- Cofactor object
            redox {int} -- Redox state of the cofactor
        Returns:
            float -- Population of the cofactor at specific redox state
        """
        cof_id = self.cofactor2id[cofactor]
        ans = 0
        for i in range(len(pop)):
            #Loop through all the possible states
            if self.idx2state(i)[cof_id] == redox:   #For every state, the number of electrons on each site is known, (ex.)state[0]=[1 2 0 3 2...], state[1]=[0 2 3 1 ...]
                # It loops through all the states to find where the cof th element of (ex.)state:[0 1 1 0 2 3...] is equal to the given redox state
                # Population of electron at each cofactor = redox state of that cofactor
                ans += pop[i]

        return ans

    def getCofactorRate(self, cof_i: Cofactor, red_i: int, cof_f: Cofactor, red_f: int, pop: np.array) -> float:
        """
        Calculate the instantaneous forward rate from cof_i to cof_f
        Arguments:
            cof_i {Cofactor} -- Cofactor object for initial cofactor
            red_i {int} -- Redox state for initial cofactor
            cof_f {Cofactor} -- Cofactor object for final cofactor
            red_f {int} -- Redox state for initial cofactor
            pop {np.array} -- Population vector      This is the population vector of the states. len(pop)=self.num_state
        Returns:
            float -- Instantaneous forward rate
        """
        cof_i_id = self.cofactor2id[cof_i]
        cof_f_id = self.cofactor2id[cof_f]
        flux = 0
        for i in range(self.num_state):
            # loop through all states, to find initial state
            if self.idx2state(i)[cof_i_id] == red_i and self.idx2state(i)[cof_f_id] == red_f - 1:
                """
                This "if" statement means: 
                "If the element that corresponds to cof_i in the state:[0 1 1 0 2 3...] is equal to the redox state of cof_i (prior to donating an electron)" and
                "If the element that corresponds to cof_f in the state:[0 1 1 0 2 3...] is equal to the (redox state of cof_f -1) (prior to accepting an electron)"
                """
                for j in range(self.num_state):
                    # loop through all states, to find final state
                    if self.idx2state(j)[cof_i_id] == red_i - 1 and self.idx2state(j)[cof_f_id] == red_f:
                        """
                        This "if" statement means: 
                        "If the element that corresponds to cof_i in the state:[0 1 1 0 2 3...] is equal to the (redox state of cof_i -1) (donated an electron)" and
                        "If the element that corresponds to cof_f in the state:[0 1 1 0 2 3...] is equal to the redox state of cof_f (accepted an electron)"
                        """
                        # initial, final state found! check other electron conservation
                        I = np.delete(self.idx2state(i), [cof_i_id, cof_f_id])
                        J = np.delete(self.idx2state(j), [cof_i_id, cof_f_id])
                        if np.array_equal(I, J):
                            # i and j state found!)
                            flux += self.K[i][j] * pop[i]      #K is rate matrix, so len(K)=self.num_state

        return flux

    def getCofactorFlux(self, cof_i: Cofactor, red_i: int, cof_f: Cofactor, red_f: int, pop: np.array) -> float:
        """
        Calculate the instantaneous NET flux from initial cofactor(state) to final cofactor(state), by calling getCofactorRate() twice
        Arguments:
            cof_i {Cofactor} -- Cofactor object for initial cofactor
            red_i {int} -- Redox state for initial cofactor
            cof_f {Cofactor} -- Cofactor object for final cofactor
            red_f {int} -- Redox state for final cofactor
            pop {np.array} -- Population vector      This is the population vector of the states. len(pop)=self.num_state
        Returns:
            float -- Instantaneous net flux
        """
        return self.getCofactorRate(cof_i, red_i, cof_f, red_f, pop) - self.getCofactorRate(cof_f, red_f, cof_i, red_i, pop)



    def getReservoirFlux(self, name: str, pop: np.array) -> float:
        """
        Calculate the instantaneous net flux into the reservoir connected to the reservoir
        Arguments:
            reservoir_id {int} -- Reservoir ID
            pop {np.array} -- Population vector      This is the population vector of the states. len(pop)=self.num_state
        Returns:
            float -- Instantaneous net flux connected to the reservoir
        """
        reservoir_id = self.reservoir2id[name]
        name, cofactor, redox_state, num_electron, deltaG, rate=self.reservoirInfo[reservoir_id]
        reverse_rate = rate * np.exp(self.beta*deltaG)
        final_redox=redox_state-num_electron      #redox_state is basically the initial redox state of the cofactor, which is info stored in reservoirInfo dict()

        return (self.population(pop, cofactor, redox_state) * rate - self.population(pop, cofactor, final_redox) * reverse_rate) * num_electron



    def getReservoirFluxMod(self, name: str, pop: np.array) -> float:
        """
        Calculate the instantaneous net flux into the reservoir connected to the reservoir
        Arguments:
            reservoir_id {int} -- Reservoir ID
            pop {np.array} -- Population vector      This is the population vector of the states. len(pop)=self.num_state
        Returns:
            float -- Instantaneous net flux connected to the reservoir
        """
        reservoir_id = self.reservoir2idmod[name]
        name, cofactor, redox_state, num_electron, deltaG, cofactor1, cofactor2, rate1, redox11, redox12, rate2, redox21, redox22, rate3, redox31, redox32, rate4, redox41, redox42 =self.reservoirInfoMod[reservoir_id]
        final_redox=redox_state-num_electron      #redox_state is basically the initial redox state of the cofactor, which is info stored in reservoirInfo dict()
        cof_id = self.cofactor2id[cofactor]
        cof1 = self.cofactor2id[cofactor1]
        cof2 = self.cofactor2id[cofactor2]
        flux = 0

        for i in range(self.num_state):
            if self.idx2state(i)[cof1] == redox11 and self.idx2state(i)[cof2] == redox12:
                rate = rate1
            elif self.idx2state(i)[cof1] == redox21 and self.idx2state(i)[cof2] == redox22:
                rate = rate2
            elif self.idx2state(i)[cof1] == redox31 and self.idx2state(i)[cof2] == redox32:
                rate = rate3
            elif self.idx2state(i)[cof1] == redox41 and self.idx2state(i)[cof2] == redox42:
                rate = rate4

            reverse_rate = rate * np.exp(self.beta*deltaG)
            if self.idx2state(i)[cof_id] == redox_state:
                flux += rate*pop[i]*num_electron
            elif self.idx2state(i)[cof_id] == final_redox:
                flux -= reverse_rate*pop[i]*num_electron
            else:
                flux += 0.0


        return flux #(self.population(pop, cofactor, redox_state) * rate - self.population(pop, cofactor, final_redox) * reverse_rate) * num_electron



    def reservoirFluxPlot(self, name: str, pop_init: np.array, t: float) -> list:
        """
        Calculate the net flux into a reservoir given its id versus time
        Arguments:
            t {float} -- Final time
            pop_init {np.array} -- Initial population vector (default: None)
            reservoir_id {int} -- Reservoir id
        Returns:
            list -- Net flux into the reservoir along a period of time
        """
        T = np.linspace(0, t, 1000)  # default spacing number: 1000
        A = []
        fluxes=[]
        for time in T:
            A=self.evolve(time, pop_init)
            flux = self.getReservoirFlux(name, A)
            fluxes.append(flux)
        plt.plot(T, fluxes)

        return fluxes



    def popPlot(self, cofactor: Cofactor, redox: int, pop_init: np.array, t: float) -> list:
        """
        Calculate the population of a given cofactor at specific redox state along a period of time
        Arguments:
            t {float} -- Final time
            pop_init {numpy.array} -- Initial population vector (default: None)
            cofactor {Cofactor} -- Cofactor object
            redox {int} -- Redox state of the cofactor
        Returns:
            list -- Population of the cofactor along a period of time
        """
        T = np.linspace(0, t, 10)  # default spacing number: 100
        A=[]
        pops=[] #population at site i
        for time in T:
            B=self.evolve(time, pop_init)
            A.append(B)
            pops.append(self.population(A[-1], cofactor, redox))
            #pops.append(self.population(pop, cofactor, redox))
        plt.plot(T, pops)

        #return pops
    def getExptvalue(self, pop: np.array, cofactor: Cofactor) -> float:
        """
        Calculate the expectation value of the number of particles at a given cofactor at a given time
        Arguments:
            cofactor {Cofactor} -- Cofactor object
            pop {cp.array} -- Population vector of the states
        """
        cof_id = self.cofactor2id[cofactor]
        expt=0
        #loop through all the possible states
        for i in range(self.num_state):
            expt+=self.idx2state(i)[cof_id]*pop[i]   #sum((number of particle)*(probability))

        return expt



    def getHeatvalue(self, pop: np.array) -> float:
        """

        """
        heat=0
        #loop through all the possible states
        for i in range(self.num_state):
            for j in range(self.num_state):
                if i != j:
                    if self.K[i][j] != 0:
                        #print('heat:', i, j, pop[i], pop[j], heat, self.K[i][j], self.K[j][i])
                        heat+=-0.5*(self.K[i][j]*pop[j]-self.K[j][i]*pop[i])*np.log(self.K[j][i]/(self.K[i][j]))
        return heat



    def getEntropyvalue(self, pop: np.array) -> float:
        """

        """
        entropy=0
        #loop through all the possible states
        for i in range(self.num_state):
            for j in range(self.num_state):
                if i != j:
                    if self.K[i][j]*pop[j] != 0:
                        #print('heat:', i, j, pop[i], pop[j], heat, self.K[i][j], self.K[j][i])
                        entropy+=-0.5*(self.K[i][j]*pop[j]-self.K[j][i]*pop[i])*np.log((self.K[j][i]*pop[i])/(self.K[i][j]*pop[j]))
        return entropy




    def getEnergyvalue(self, pop:np.array) -> float:
        Evalue = 0.0
        for i in range(self.num_state):
            Evalue += pop[i]*self.Elist[i]
        return Evalue



