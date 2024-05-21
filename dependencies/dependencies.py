# needed libraries
import numpy as np
from scipy.integrate import quad
import cmath
from collections import OrderedDict
import itertools
from scipy import linalg
from numpy import linalg as LA
from scipy.linalg import sqrtm
import math
from scipy import sparse
import collections
from collections import Counter
from collections import defaultdict

import random



############################# DEFINING THE HAMILTONIANS ##########################


def tfim_LocalHamiltonian(J,h):
    """     
    From: ' Analysis of the transverse field Ising model by continuous unitary transformations. 
    Master Thesis, Lehrstuhl für Theoretische Physik I
    Fakultät Physik
    Technische Universität Dortmund
    Benedikt Fauseweh
    2012 
    """

    sX, sY, sZ, sI = PauliMatrice()

    hloc = (-(J/4.0)*np.kron(sX, sX) + (h/2.0)*np.kron(sZ,sI) ).reshape(2, 2, 2, 2)

    return hloc


def tfim_LocalHamiltonian_new(lambdaa):

    """ Trasverse Ising Hamiltonian - Fabio/Marco """ 

    sX, sY, sZ, sI = PauliMatrice()

    #hloc = (-lambdaa*np.kron(sX, sX) - 0.5*(np.kron(sZ,sI) + np.kron(sI,sZ)) ).reshape(2, 2, 2, 2)
    hloc = (lambdaa*np.kron(sX, sX) - np.kron(sZ,sI) ) .reshape(2, 2, 2, 2)
    return hloc    

def tfim_test(J, h):

    """ Trasverse Ising Hamiltonian - Fabio/Marco """ 

    sX, sY, sZ, sI = PauliMatrice()

    #hloc = (J*np.kron(sX, sX) - h*0.5*(np.kron(sZ,sI) + np.kron(sI,sZ) ) ) .reshape(2, 2, 2, 2)

    #sX = matrixexponential(1j*(np.pi/8.0)*sY) @ sX @ matrixexponential(-1j*(np.pi/8.0)*sY)
    #sZ = matrixexponential(1j*(np.pi/8.0)*sY) @ sZ @ matrixexponential(-1j*(np.pi/8.0)*sY)


    hloc = (J*np.kron(sX, sX) - h*0.5*(np.kron(sZ,sI) + np.kron(sI,sZ) ) ) .reshape(2, 2, 2, 2)

    return hloc 

def tfim_test1(J, h):

    """ Trasverse Ising Hamiltonian - Fabio/Marco """ 

    sX, sY, sZ, sI = PauliMatrice()

    #hloc = (J*np.kron(sX, sX) - h*0.5*(np.kron(sZ,sI) + np.kron(sI,sZ) ) ) .reshape(2, 2, 2, 2)

    sX = matrixexponential(1j*(np.pi/8.0)*sY) @ sX @ matrixexponential(-1j*(np.pi/8.0)*sY)
    sZ = matrixexponential(1j*(np.pi/8.0)*sY) @ sZ @ matrixexponential(-1j*(np.pi/8.0)*sY)


    hloc = (J*np.kron(sX, sX) - h*0.5*(np.kron(sZ,sI) + np.kron(sI,sZ) ) ) .reshape(2, 2, 2, 2)

    return hloc 


    

def XYchain_LocalHamiltonian(lambdaa):

    """ XY chain Hamiltonian """ 

    sX, sY, sZ, sI = PauliMatrice()

    
    hloc = (0.5*np.kron(sX, sX) + lambdaa*np.kron(sY,sY)) .reshape(2, 2, 2, 2)
    return hloc    


def XYZchain_LocalHamiltonian(Jx,Jy,Jz,h):

    """ XY chain Hamiltonian """ 

    sX, sY, sZ, sI = PauliMatrice()

    
    hloc = (Jx*np.kron(sX, sX) + Jy*np.kron(sY,sY) + Jz*np.kron(sZ,sZ) - h*0.5*(np.kron(sI,sZ) + np.kron(sZ,sI))) .reshape(2, 2, 2, 2)
    return hloc    



def XYZchain_LatticeTranslationOperator(N):

    sX, sY, sZ, sI = PauliMatrice()

    # operator indexing; 0: identity, 1: x opeartor, 2: y operator, 3: z operator 

    T = 0.5*(construct_FullHilbert(0,0,N-1,N,N) + construct_FullHilbert(1,1,N-1,N,N) + construct_FullHilbert(2,2,N-1,N,N) + construct_FullHilbert(3,3,N-1,N,N))
    Tbasic = T.copy()


    for i in range(N-1,1,-1):

        Ttampon = 0.5*(construct_FullHilbert(0,0,i-1,i,N) + construct_FullHilbert(1,1,i-1,i,N) + construct_FullHilbert(2,2,i-1,i,N) + construct_FullHilbert(3,3,i-1,i,N))

        Tbasic = Ttampon @ Tbasic




    return Tbasic


def construct_FullHilbert(operator1, operator2, position1, position2, totalsize):

    sX, sY, sZ, sI = PauliMatrice()


    gate_list = [sI, sX, sY, sZ]
    gate_list_id = [0, 1, 2, 3]


    temp = np.array(np.zeros(totalsize))
    temp[position1 - 1] = int(operator1)
    temp[position2 - 1] = int(operator2)

    #print('tempaaaa =', temp,temp[0])

    gate = gate_list[int(temp[0])]
    for j in range(1,len(temp)):
        gate = np.kron(gate,gate_list[int(temp[j])])

    full_operator = gate.copy()

    return full_operator


def XYchainDM_LocalHamiltonian(J,gamma,D):

    """ Dzyalozinski- Morya Hamiltonian - Petar Mali paper"""

    sX, sY, sZ, sI = PauliMatrice()

    hloc = ((J/4.0)*(1.0 + gamma)*np.kron(sX, sX) - (J/4.0)*(1.0 - gamma)*np.kron(sY, sY) +  ((D*J)/4.0)*(np.kron(sX, sY) + np.kron(sY, sX))).reshape(2, 2, 2, 2)

    return hloc

def FabioMarcoFrustrated_LocalHamiltonian(delta,phi):

    """ Fabio/Marco Hamiltonian"""


    sX, sY, sZ, sI = PauliMatrice()

    local_parity_operator = (np.kron(sI,sI) + np.kron(sZ,sZ) )

    hloc = (0.25*local_parity_operator @ ((np.cos(delta)*(np.cos(phi)*np.kron(sX,sX) + np.sin(phi)*np.kron(sY,sY)) - np.sin(delta)*np.kron(sZ,sZ))) @ local_parity_operator).reshape(2,2,2,2)

    return hloc


#########################  THE REST OF THE FUNCTIONS USED IN THE CODES ####################

def doApplyHamTENSOR(psi: np.ndarray, h: np.ndarray, N: int, usePBC: bool):
    """ Basic succesive application of the local 4 leg local Hamiltonian to the psi tensor. 
        This routine is designed to work with the LinearOperator function and the Lanczos type algorithms.
        Args:
            psi:        vector of length d**N describing the quantum state.
            h:          array of ndim=4 describing the nearest neighbor coupling (local Hamiltonian of dimension (d^2, d^2)).
            N:          the number of lattice sites.
            usePBC:     sets whether to include periodic boundary term.
        Returns:
            np.ndarray: state psi after application of the Hamiltonian.

        Logic of the application is explained somewhere in the notes. 
        First, second and last application of the local Hamiltonian are written explicelty. The remaining steps are generic and the same algorithm is applied.      
    """
    # this is the local dimension 
    d = 2

    # reshape the vector into a tensor with N legs each with dimension d
    reshape_array_default     = np.full(N,d)
    psi = psi.reshape(reshape_array_default)

    # First application of the local Hamiltonian to the tensor quantum state for legs 1 and 2 (k = 0)
    psiOut = (h.reshape(4,4) @ psi.reshape(4,2**(N-2))).reshape(reshape_array_default).reshape(2**N)


    # Second application of the local Hamiltonian to the tensor quantum state for legs 2 and 3 (k = 1)
    # generate the lists of transposition (first transpose different from the second one)
    transpose_array_default1    = np.arange(N)
    transpose_array_default2    = transpose_array_default1.copy()

    transpose_array_default1[0] = 1
    transpose_array_default1[1] = 2
    transpose_array_default1[2] = 0 

    transpose_array_default2[0] = 2
    transpose_array_default2[1] = 0
    transpose_array_default2[2] = 1 

    psiOut += (h.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)


    # Remaining applications of the local Hamiltonian  (k >= 2)
    for k in range(2,N-1):

        transpose_array_default         = np.arange(N)
        transpose_array_default[0]      = k
        transpose_array_default[1]      = k+1
        transpose_array_default[k]      = 0 
        transpose_array_default[k+1]    = 1

        psiOut += (h.reshape(4,4) @ psi.transpose(transpose_array_default).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)

        
    # PBC conditions application of the local hamiltonian on the end tails of the psi tensor
    if usePBC:

        # generate the lists of transposition (first transpose different from the second one)
        transpose_array_default1  = np.arange(N)
        transpose_array_default2  = transpose_array_default1.copy()

        transpose_array_default1[0]    = N - 1
        transpose_array_default1[1]    = 0
        transpose_array_default1[N-1]  = 1

        transpose_array_default2[0]    = 1
        transpose_array_default2[1]    = N-1
        transpose_array_default2[N-1]  = 0

        psiOut += (h.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)


    return psiOut


def local_magnetization_single_point(sigma,j,psi,N,d):
    """ function to evaluate single point correlator < psi | sigma_{j}^{x,y,z} | psi > - magnetization """ 

    initial_psi = psi.copy()

    # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default     = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)

    # transpose the tensor with N legs to match the position of the local operator
    transpose_array_default        = np.arange(N)
    transpose_array_default[0]     = j-1
    transpose_array_default[j-1]   = 0

    psi = (sigma @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)

    mag = (np.transpose(np.conjugate(initial_psi))) @ psi

    return mag  

def local_magnetization_two_point(sigma,j,t,psi,N,d):
    """ # function to evaluate two point spin correlator < psi | sigma_{j + t}^{x,y,z} sigma_{j}^{x,y,z} | psi > - ONLY MADE TO WORK WITH PBC IMPOSED ON THE PROBLEM (see later mod function % which loops around psi) """

    initial_psi = psi.copy()

    # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)

    # transpose the tensor with N legs to match the position of the local operator
    transpose_array_default       = np.arange(N)
    transpose_array_default[0]    = j-1
    transpose_array_default[j-1]  = 0

    # first operator application
    psi = (sigma @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default)

    # transpose the tensor with N legs to match the position of the local operator
    transpose_array_default       = np.arange(N)
    transpose_array_default[0]    = (j+t-1) % N               
    transpose_array_default[(j+t-1) % N]  = 0


    # second operator application
    psi = (sigma @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)

    mag = np.transpose(initial_psi) @ psi

    return mag  


def local_magnetization_two_point_general(sigma2,sigma1,j,t,psi,N,d):
    """ # function to evaluate two point spin correlator < psi | sigma_{j + t}^{x,y,z} sigma_{j}^{x,y,z} | psi > - ONLY MADE TO WORK WITH PBC IMPOSED ON THE PROBLEM (see later mod function % which loops around psi) """

    initial_psi = psi.copy()

    # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)

    # transpose the tensor with N legs to match the position of the local operator
    transpose_array_default       = np.arange(N)
    transpose_array_default[0]    = j-1
    transpose_array_default[j-1]  = 0

    # first operator application
    psi = (sigma1 @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default)

    # transpose the tensor with N legs to match the position of the local operator
    transpose_array_default       = np.arange(N)
    transpose_array_default[0]    = t-1               
    transpose_array_default[t-1]  = 0


    # second operator application
    psi = (sigma2 @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)

    mag = np.transpose(initial_psi) @ psi

    return mag  

def local_magnetization_general_point_general(sigma,jlist,psi,N,d):
    """ # function to evaluate two point spin correlator < psi | sigma_{j1}^{x,y,z} sigma_{j2}^{x,y,z} ... sigma_{jN}^{x,y,z}| psi > - ONLY MADE TO WORK WITH PBC IMPOSED ON THE PROBLEM (see later mod function % which loops around psi) """

    initial_psi = psi.copy()

    # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)

    # general loop
    for ii in range(0,len(jlist)):
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default       = np.arange(N)
        transpose_array_default[0]    = jlist[ii]-1
        transpose_array_default[jlist[ii]-1]  = 0

        # first operator application
        psi = (sigma[ii] @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default)

    psi = psi.reshape(2**N)

    mag = np.transpose(initial_psi) @ psi

    return mag  


def local_magnetization_two_neighboring_point(sigma,j,psi,N,d):
    """ this is the function which computes < psi | sigma_{j}^{x,y,z} sigma_{j+1}^{x,y,z} | psi > - this should give the same output as the function :local_magnetization_two_point: for t = 1   """

    sigma = np.kron(sigma, sigma)

    initial_psi = psi.copy()

    # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)


    if (j == N):
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default1       = np.arange(N)
        transpose_array_default2       = transpose_array_default1.copy()

        transpose_array_default1[0]    = N-1
        transpose_array_default1[1]    = 0
        transpose_array_default1[N-1]  = 1

        transpose_array_default2[0]    = 1
        transpose_array_default2[1]    = N-1
        transpose_array_default2[N-1]  = 0

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)

    elif(j == 2):

        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default1       = np.arange(N)
        transpose_array_default2       = transpose_array_default1.copy()

        transpose_array_default1[0]    = 1
        transpose_array_default1[1]    = 2
        transpose_array_default1[2]    = 0

        transpose_array_default2[0]    = 2
        transpose_array_default2[1]    = 0
        transpose_array_default2[2]    = 1

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)
      
    else:    
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default       = np.arange(N)
        transpose_array_default[0]    = j-1
        transpose_array_default[1]    = j
        transpose_array_default[j-1]  = 0
        transpose_array_default[j]    = 1


        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)

    mag = np.transpose(initial_psi) @ psi


    return mag

def tfim_ground_state_energy_per_site_exact(N,J,h):
    """
    Exact results in Analysis of the transverse field Ising model by continuous unitary transformations. 
    Master Thesis, Lehrstuhl für Theoretische Physik I
    Fakultät Physik
    Technische Universität Dortmund
    Benedikt Fauseweh
    2012 """

    integral = quad(tfim_ground_state_energy_per_site_exact_integrand, 0, np.pi, args=(J, h))[0]
    return -(h/(2.0*np.pi))*integral

def tfim_ground_state_energy_per_site_exact_integrand(q, J, h):
    return (cmath.sqrt(1.0 + (J**2)/(4*h**2) - (J*cmath.cos(q))/(h) )).real


def tfim_ground_state_magnetizationSigmaZ_per_site_exact(N,J,h):
    """
    Exact results in Analysis of the transverse field Ising model by continuous unitary transformations. 
    Master Thesis, Lehrstuhl für Theoretische Physik I
    Fakultät Physik
    Technische Universität Dortmund
    Benedikt Fauseweh
    2012 """

    integral = quad(tfim_ground_state_magnetization_per_site_exact_integrand, 0, np.pi, args=(J, h))[0]
    return (1.0/(np.pi))*integral

def tfim_ground_state_magnetization_per_site_exact_integrand(q, J, h):

    return ((1.0 - (J*cmath.cos(q))/(2.0*h))/((cmath.sqrt(1.0 + (J**2)/(4*h**2) - (J*cmath.cos(q))/(h) )))).real


def tfim_ground_state_magnetizationSigmaX_per_site_exact(N,J,h):
    """
    Exact results in Analysis of the transverse field Ising model by continuous unitary transformations. 
    Master Thesis, Lehrstuhl für Theoretische Physik I
    Fakultät Physik
    Technische Universität Dortmund
    Benedikt Fauseweh
    2012 """

    if (J >= 2.0*h):
        value = (1.0 - (4.0*h**2)/(J**2))**(1.0/8.0)

    if (J < 2.0*h):
        value = 0.0


    return value


def PartialTraceGeneralTensor(N,index_list,A):
    """ Function that computes the partial trace over index_list indices (the index list needs to be ordered from smaller to bigger index)

 


    """

    # reshape the input vectors into tensors (here we exploit the fact that psi* is just the complex conjugate of psi )
    reshape_array_default = np.full(N,2)    
    A_initial = A.reshape(reshape_array_default)

    # generate initial transpose indices vector (we apply permutations and operatorion so transposition is correctly performed )
    list_A = np.arange(N)
    list_B = np.arange(N)

    # this changing the indeces by one is because of python stuff (the numbering starts from zero and not 1)
    index_list = np.array(index_list) - 1

    ##### generating the first transpose rule for A ###

    ## initial step of moving the indices to the left
    for zz in range(0,len(index_list)):
        list_A[zz] = index_list[zz]

    ## figure out what are the missing indices that happen because of overwritting in loop above
    list_A_no_dupl = list(OrderedDict.fromkeys(list_A))
    missing_indices = np.delete(np.arange(N), list_A_no_dupl)

    ## now replace the doubled indices with indices in the missing_indices array
    counter = 0
    for zz in range(len(index_list),len(list_A)):
        for tt in range(0,len(index_list)):
            if (list_A[zz] == index_list[tt]):
                list_A[zz] = missing_indices[counter]
                counter += 1
    

    ##### generating the first transpose rule for B ###

    ## initial step of moving the indices to the right
    for zz in range(0,len(index_list)):
        list_B[len(list_B) - zz - 1] = index_list[len(index_list) - zz - 1]

    ## figure out what are the missing indices that happen because of overwritting in loop above
    list_B_no_dupl = list(OrderedDict.fromkeys(list_B))
    missing_indices = np.delete(np.arange(N), list_B_no_dupl)

    ## now replace the doubled indices with indices in the missing_indices array
    counter = 0
    for zz in range(0,len(list_B)-len(index_list)):
        for tt in range(0,len(index_list)):
            if (list_B[zz] == index_list[tt]):
                list_B[zz] = missing_indices[counter]
                counter += 1


    ##### generating the second transpose rule for A ###

    list_A_cut = list_A[len(index_list):]
    list_A_cut_sort = np.sort(list_A_cut)

    list_B_cut = list_B[:-len(index_list)]
    list_B_cut_sort = np.sort(list_B_cut)

    transpose2_A = np.append(index_list,list_A_cut_sort)
    transpose2_B = np.append(list_B_cut_sort,index_list)

    ############### MAIN OPERATION AFTER ALL PREPARATION HAS BEEN PERFORMED ::: TRANSPOSITION ON A and B
    A = A_initial.transpose(transpose2_A).reshape(2**len(index_list),2**(N - len(index_list)))
    B = A_initial.transpose(transpose2_B).reshape(2**(N - len(index_list)),2**len(index_list))

    # FINAL MULTIPLICATION
    out = (A @ np.conjugate(B))

    return out


def PauliMatrice():

    """ Defining the Pauli matrices """

    sX = np.array([[0, 1.0], [1.0, 0]])
    sY = np.array([[0, -1.0j], [1.0j, 0]])
    sZ = np.array([[1.0, 0], [0, -1.0]])
    sI = np.array([[1.0, 0], [0, 1.0]])

    return sX,sY,sZ,sI

def PartialTraceGeneralCorrelation(N,index_list,A,B):
    """ Evalute the Partial Trace by evaluating spin point correlations function. This was used as a check. It turns out that evalauting concurrence is much faster anyway using direct partial trace when in tensor representation. """
    
    reverse_index_list = index_list[::-1]

    # load in the Pauli Matrices
    sX, sY, sZ, sI = PauliMatrice()

    # generate 
    x = [sI,sX,sY,sZ]
    list_of_2point_correlatores = [p for p in itertools.product(x, repeat=len(index_list))]

    value = np.zeros((2,2),dtype=complex)
    for ii in range(0,len(index_list)-1):
        value = np.kron(value, np.zeros((2,2),dtype=complex) )


    for i in range(0,len(list_of_2point_correlatores)):


        # generate the list of operators that needs to be evalued
        sigmalist = []
        for ii in range(0,len(index_list)):
            sigmalist.append(list_of_2point_correlatores[i][ii])

        # generate the tensor product of those same operators in the list above
        kkron = list_of_2point_correlatores[i][0]
        for ii in range(1,len(index_list)):
                kkron = np.kron(kkron,list_of_2point_correlatores[i][ii] )


        value = value + local_magnetization_general_point_general(sigmalist,reverse_index_list,psi,N,2)*kkron


    return value

def local_magnetization_two_point_general(sigma2,sigma1,j,t,psi,N,d):
    """  function to evaluate two point spin correlator < psi | sigma_{j + t}^{x,y,z} sigma_{j}^{x,y,z} | psi > - ONLY MADE TO WORK WITH PBC IMPOSED ON THE PROBLEM (see later mod function % which loops around psi) """

    initial_psi = psi.copy()

    # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)

    # transpose the tensor with N legs to match the position of the local operator
    transpose_array_default       = np.arange(N)
    transpose_array_default[0]    = j-1
    transpose_array_default[j-1]  = 0

    # first operator application
    psi = (sigma1 @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default)

    # transpose the tensor with N legs to match the position of the local operator
    transpose_array_default       = np.arange(N)
    transpose_array_default[0]    = t-1               
    transpose_array_default[t-1]  = 0


    # second operator application
    psi = (sigma2 @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)

    mag = np.transpose(initial_psi) @ psi

    return mag  



def ApplyLocalGate(sigma,index_pair,psi,N,d,dt):
    """ apply exponentiated local 2 site gate, this is generalized to any distance pair but does not involve pairs that are there due to PBC"""

    sigma = linalg.expm(-1j*dt*sigma)

    j = index_pair[0]

   # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)

    if (j == N):
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default1       = np.arange(N)
        transpose_array_default2       = transpose_array_default1.copy()

        transpose_array_default1[0]    = N-1
        transpose_array_default1[1]    = 0
        transpose_array_default1[N-1]  = 1

        transpose_array_default2[0]    = 1
        transpose_array_default2[1]    = N-1
        transpose_array_default2[N-1]  = 0

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)


    elif(j == 2):

        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default1       = np.arange(N)
        transpose_array_default2       = transpose_array_default1.copy()

        transpose_array_default1[0]    = 1
        transpose_array_default1[1]    = 2
        transpose_array_default1[2]    = 0

        transpose_array_default2[0]    = 2
        transpose_array_default2[1]    = 0
        transpose_array_default2[2]    = 1

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)
      
    else:    
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default       = np.arange(N)
        transpose_array_default[0]    = j-1
        transpose_array_default[1]    = j
        transpose_array_default[j-1]  = 0
        transpose_array_default[j]    = 1

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)


    return psi

def ApplyLocalGate_GENERAL(sigma,index_pair,psi,N,d,dt):
    """ apply exponentiated local 2 site gate, this is generalized to any distance pair but does not involve pairs that are there due to PBC. THIS IS GENERAL BECAUSE WE DO NOT EXPONENTIATE THE HERMITIAN MATRICES INSIDE"""


    j = index_pair[0]

   # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)

    if (j == N):
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default1       = np.arange(N)
        transpose_array_default2       = transpose_array_default1.copy()

        transpose_array_default1[0]    = N-1
        transpose_array_default1[1]    = 0
        transpose_array_default1[N-1]  = 1

        transpose_array_default2[0]    = 1
        transpose_array_default2[1]    = N-1
        transpose_array_default2[N-1]  = 0

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)


    elif(j == 2):

        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default1       = np.arange(N)
        transpose_array_default2       = transpose_array_default1.copy()

        transpose_array_default1[0]    = 1
        transpose_array_default1[1]    = 2
        transpose_array_default1[2]    = 0

        transpose_array_default2[0]    = 2
        transpose_array_default2[1]    = 0
        transpose_array_default2[2]    = 1

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)
      
    else:    
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default       = np.arange(N)
        transpose_array_default[0]    = j-1
        transpose_array_default[1]    = j
        transpose_array_default[j-1]  = 0
        transpose_array_default[j]    = 1

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)


    return psi



def concurrence_measure(rho):
    """ computing the concurrence measure of entanglement """

    sX, sY, sZ, sI = PauliMatrice()

    rho_sqrt = sqrtm(rho)

    rho_fin =  sqrtm( rho_sqrt @ (np.kron(sY,sY) @ np.conjugate(rho) @ np.kron(sY,sY)) @ rho_sqrt )


    eigenvalues,eigenvectors = LA.eigh(rho_fin)
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]

    con =  np.max([eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3], 0.0])


    return con


def generate_dictionary(N :int,r :int, usePBC :bool):
    """ generate the list of pairs for the application of the local Hamiltonian  """
    lista = []

    for iii in range(0,N-r):
        lista.append([iii+1,iii+1+r])


    if usePBC:
        for iii in range(0,r):
            lista.append([N-iii,r-iii])



    return lista



def magnetizationSigmaZ_per_site_exact(lambdaa):
    """
    
    magnetization exact result

    PHYSICAL REVIEW B 78, 224413 共2008兲


    """

    integral = quad(ground_state_magnetization_per_site_exact_integrand, 0, np.pi, args=(lambdaa))[0]
    return -(1.0/(np.pi))*integral

def ground_state_magnetization_per_site_exact_integrand(q, lambdaa):
    """" magnetizaion exact result integrand under the integral """
    return ( (1.0 + lambdaa*np.cos(q) )/(np.sqrt( (lambdaa*np.sin(q))**2.0 + (1.0 + lambdaa*np.cos(q))**2.0 ) ) ).real


def select_cooling_evolution_gates():
    """ defines a set of evolution gates from which we choose one random uniformly with an equal probability """
    # define Pauli matrices
    sX, sY, sZ, sI = PauliMatrice()

    # define the 3 'single' and 3 two body operators
    single_body1 =  np.kron(sX,sI) + np.kron(sI,sX)
    single_body2 =  np.kron(sY,sI) + np.kron(sI,sY)
    single_body3 =  np.kron(sZ,sI) + np.kron(sI,sZ)

    two_body1 = np.kron(sX,sX)
    two_body2 = np.kron(sY,sY)
    two_body3 = np.kron(sZ,sZ)

    gate_list = [single_body1,single_body2,single_body3,two_body1,two_body2,two_body3]

    # select a random values (uniform)
    rand_value = np.random.uniform(0,1,1)

    idd = 0
    # take the gate from the gate_list with equal probability
    for ij in range(0,6):
        if (rand_value <= (1.0/6.0)*(ij + 1)):
            if (rand_value > (1.0/6.0)*(ij)):
                gate = gate_list[ij]
                idd = ij + 1



    return gate,idd



def select_cooling_evolution_gates_GENERAL(dt):
    """ defines a set of evolution gates from which we choose one random uniformly with an equal probability """
    # define Pauli matrices
    sX, sY, sZ, sI = PauliMatrice()


    two_body1 = np.kron(sX,sX)
    two_body2 = np.kron(sY,sY)
    two_body3 = np.kron(sZ,sZ)

    gate_list = [two_body1,two_body2,two_body3]

    # select a random values (uniform)
    rand_value = np.random.uniform(0,1,1)

    idd = 0
    # take the gate from the gate_list with equal probability
    for ij in range(0,3):
        if (rand_value <= (1.0/3.0)*(ij + 1)):
            if (rand_value > (1.0/3.0)*(ij)):
                gate = gate_list[ij]
                idd = ij + 1

    gate = linalg.expm(-1j*dt*gate)

    return gate,idd

def select_cooling_evolution_indices(list_pairs):
    """ select a random pair of indices (adjacent) to which we apply to the local gates later """

    # select a random value (uniform)
    rand_value = np.random.uniform(0,1,1)

    # take the random pair with equal probability
    for ij in range(0,len(list_pairs)):
        if (rand_value <= (1.0/len(list_pairs))*(ij + 1)):
            if (rand_value > (1.0/len(list_pairs))*(ij)):
                pair = list_pairs[ij]

    # find where the randomly selected pair is located in the initial list of pairs
    idd =  list_pairs.index(pair)


    return pair, idd


def concurrence_from_state(Nsites,psi,whichone):
    """ compute the concurrence from a given state at distance r """
    r = whichone

   # generate the list of pairs for concurence measurement
    list_pairs = generate_dictionary(Nsites,r, True)

    # compute the average concurrence
    conn11 = []
    for jj in range(0,len(list_pairs)):

        rho = PartialTraceGeneralTensor(Nsites,list_pairs[jj],psi) 

        conn11.append(concurrence_measure(rho))

    concurrence = np.average(conn11)


    return concurrence

def concurrence_from_state_summ(Nsites,psi,upto):
    """ compute the concurrence from a givn state, but compute and add up nearest, next, next-to... nearest neighbors (r = 1, 2, 3,... upto) into a single number for each of the possible pairings at each level"""

    # loop over the r's
    add = 0
    for kk in range(1,upto + 1 ):
        # generate the list of pairs for concurence measurement
        list_pairs = generate_dictionary(Nsites,kk, True)

        # compute the average concurrence
        conn11 = []
        for jj in range(0,len(list_pairs)):
            rho = PartialTraceGeneralTensor(Nsites,list_pairs[jj],psi) 

            conn11.append(concurrence_measure(rho))


        add = add + np.average(conn11)

    concurrence = add


    return concurrence



def vonNeumann(N,R,psi):
    """ compute the von Neumann entropy of a state psi with the subsystem being size R  """

    # fix the R 
    R = int(math.floor(R))
    #R = int(math.ceil(R))
   
    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)

    ee = []
    for i in range(0,len(list_indic)):
        # compute the partial trace
        rho_reduced = PartialTraceGeneralTensor(N,list_indic[i],psi) 

        # compute eigenvalues
        eigenvalues,eigenvectors = LA.eigh(rho_reduced)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]

        # compute the von Neumann entropy
        entropy = 0.0
        epsi = 10**(-15)

        for ii in range(0,len(eigenvalues)):
            if (eigenvalues[ii] > epsi):
                entropy += eigenvalues[ii]*np.log2(eigenvalues[ii])

#        print("entropy------------------------------------------")
 #       print(entropy)
        ee.append(entropy)        


    entropy = - np.average(ee)


    return entropy,ee

def vonNeumann_natbase(N,R,psi):
    """ compute the von Neumann entropy of a state psi with the subsystem being size R  """

    # fix the R 
    R = int(math.floor(R))
    #R = int(math.ceil(R))
   
    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)

    ee = []
    for i in range(0,len(list_indic)):
        # compute the partial trace
        rho_reduced = PartialTraceGeneralTensor(N,list_indic[i],psi) 

        # compute eigenvalues
        eigenvalues,eigenvectors = LA.eigh(rho_reduced)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]

        # compute the von Neumann entropy
        entropy = 0.0
        epsi = 10**(-15)

        for ii in range(0,len(eigenvalues)):
            if (eigenvalues[ii] > epsi):
                entropy += eigenvalues[ii]*np.log(eigenvalues[ii])

#        print("entropy------------------------------------------")
 #       print(entropy)
        ee.append(entropy)        


    entropy = - np.average(ee)


    return entropy,ee


def vonNeumann_aftergate(N,R,psi,gate_id):
    """ compute the von Neumann entropy of a state psi with the subsystem being size R. This function is different from vonNeumann because it computes only 2 of the partitions that are actually relevant after the application of a local gate. Variable gate_id provides this information.  """

    # fix the R 
    R = int(math.floor(R))
    #R = int(math.ceil(R))
   
    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)


    kk = []
    relevant_list = []
    for i in range(0,len(list_indic)):
        
        if (list_indic[i][0] == gate_id[1]):
            kk.append(list_indic[i])

        if (list_indic[i][R-1] == gate_id[0]):
            kk.append(list_indic[i])    




    ee = []
    for i in range(0,len(kk)):
        # compute the partial trace
        rho_reduced = PartialTraceGeneralTensor(N,kk[i],psi) 

        # compute eigenvalues
        eigenvalues,eigenvectors = LA.eigh(rho_reduced)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]

        # compute the von Neumann entropy
        entropy = 0.0
        epsi = 10**(-15)


        for ii in range(0,len(eigenvalues)):
            if (eigenvalues[ii] > epsi):
                entropy += eigenvalues[ii]*np.log2(eigenvalues[ii])

        ee.append(entropy)        

    entropy = - np.average(ee)


    return entropy,ee


def vonNeumann_aftergate_correct(N,R,psi,gate_id):
    """ compute the von Neumann entropy of a state psi with the subsystem being size R. This function is different from vonNeumann because it computes only 2 of the partitions that are actually relevant after the application of a local gate. Variable gate_id provides this information.  """

    # fix the R 
    R = int(math.floor(R))
    #R = int(math.ceil(R))
   
    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)


    kk = []
    relevant_list = []
    for i in range(0,len(list_indic)):
        
        if (list_indic[i][0] == gate_id[1]):
            kk.append(list_indic[i])
            #print(list_indic[i],i)
            relevant_list.append(i)
        if (list_indic[i][R-1] == gate_id[0]):
            kk.append(list_indic[i])    
            #print(list_indic[i],i)
            relevant_list.append(i)



    ee = []
    for i in range(0,len(kk)):
        # compute the partial trace
        rho_reduced = PartialTraceGeneralTensor(N,kk[i],psi) 

        # compute eigenvalues
        eigenvalues,eigenvectors = LA.eigh(rho_reduced)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]

        # compute the von Neumann entropy
        entropy = 0.0
        epsi = 10**(-15)

        #print('hahaa = ',len(eigenvalues))


        for ii in range(0,len(eigenvalues)):

            if (eigenvalues[ii] > epsi):
                #print(ii,eigenvalues[ii])
                entropy += eigenvalues[ii]*np.log2(eigenvalues[ii])

        ee.append(entropy)        

    entropy = - np.average(ee)


    return entropy,ee,relevant_list


def vonNeumann_aftergate_correct_natbase(N,R,psi,gate_id):
    """ compute the von Neumann entropy of a state psi with the subsystem being size R. This function is different from vonNeumann because it computes only 2 of the partitions that are actually relevant after the application of a local gate. Variable gate_id provides this information.  """

    # fix the R 
    R = int(math.floor(R))
    #R = int(math.ceil(R))
   
    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)


    kk = []
    relevant_list = []
    for i in range(0,len(list_indic)):
        
        if (list_indic[i][0] == gate_id[1]):
            kk.append(list_indic[i])
            #print(list_indic[i],i)
            relevant_list.append(i)
        if (list_indic[i][R-1] == gate_id[0]):
            kk.append(list_indic[i])    
            #print(list_indic[i],i)
            relevant_list.append(i)



    ee = []
    for i in range(0,len(kk)):
        # compute the partial trace
        rho_reduced = PartialTraceGeneralTensor(N,kk[i],psi) 

        # compute eigenvalues
        eigenvalues,eigenvectors = LA.eigh(rho_reduced)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]

        # compute the von Neumann entropy
        entropy = 0.0
        epsi = 10**(-15)

        #print('hahaa = ',len(eigenvalues))


        for ii in range(0,len(eigenvalues)):

            if (eigenvalues[ii] > epsi):
                #print(ii,eigenvalues[ii])
                entropy += eigenvalues[ii]*np.log(eigenvalues[ii])

        ee.append(entropy)        

    entropy = - np.average(ee)


    return entropy,ee,relevant_list

def generate_dictionary_adjacent(N :int, r :int):
    """ generate the list used evaluate of the von Neuman entropy, and only adjacent spins are to be considered  """

    lista = []
    r = r - 1 
    for iii in range(0,N):
        ll_tampon = [(iii ) % (N)]
        for jjj in range(0,r):
            ll_tampon.append((iii+jjj+1) % (N))
    
        for ii in range(0,len(ll_tampon)):
            ll_tampon[ii] += 1 
        lista.append(ll_tampon)


    return lista    


def func_for_fit(x, a, b, c):
    """ Fitting function for the concurrence in the odd case. """ 
    return a*np.array(x)**b  + c

def func_for_fit_old(x, a, b):
    """ Fitting function for the concurrence in the odd case. """ 
    return a*np.array(x)**b




def U(dt :float):
    """ Generating 2x2 random unitary matrix (from 1. Zyczkowski K, Kus M. Random unitary matrices. J Phys A: Math Gen [Internet]. 1994 Jun [cited 2021 Mar 26];27(12):4235–45. ) """ 
    alpha, psi, chi = np.random.uniform(low=0.0, high=2*np.pi, size=3)
    phi = np.arcsin(np.sqrt(np.random.uniform(low=0.0, high=1.0)))

    res = np.zeros((2,2),dtype=np.complex64)
    res[0,0] = np.exp(1.0j*psi)*np.cos(phi)
    res[0,1] = np.exp(1.0j*chi)*np.sin(phi)
    res[1,0] = -np.exp(-1.0j*chi)*np.sin(phi)
    res[1,1] = np.exp(-1.0j*psi)*np.cos(phi)

    return np.exp(1.0j*alpha) * res



def random_gate(dt :float):
    """ the total 4x4 sized random unitary actually is constructed from a tensor product of two 2x2 tensors multiplied with a tensor of 4x4 which is constructed directly from the three nontricial gates sigma_x sigma_x, sigma_y sigma_y, sigma_z sigma_z"""
    U1 = U(dt)
    U2 = U(dt)

    ridx = np.random.randint(3)

    sX, sY, sZ, sI = PauliMatrice()


    if (ridx == 0):
        Uxx = linalg.expm(1.0j*dt*np.kron(sX,sX))
        return np.kron(U1, U2) @ Uxx, ridx
    elif (ridx == 1):
        Uyy = linalg.expm(1.0j*dt*np.kron(sY,sY))
        return np.kron(U1, U2) @ Uyy, ridx
    else:
        Uzz = linalg.expm(1.0j*dt*np.kron(sZ,sZ))
        return np.kron(U1, U2) @ Uzz, ridx

def wignersurmiseGUE(x):
	""" Wigner surmise for GUE - just expression"""
	return ((32.0*x*x)/(np.pi*np.pi))*np.exp(-(4.0/np.pi)*x*x)



def histogram_generate(bins,array):						
	""" generating a line from a histogram data, so given a 1d array of data this function creates a line plot data """
	minarray = np.min(array)
	maxarray = np.max(array)
	totalsize = abs(minarray) + abs(maxarray)
	binsize = totalsize/bins
	
	
	# defining the histogram array
	histogramcounter = []
	for i in range(0,bins):
		histogramcounter.append(0)

	# filling in the histogram array
	for i in range(0,len(array)):
		#print(i)
		for j in range(0,bins):
			if (array[i] >= minarray + j*binsize):
				if (array[i] <= minarray + (j+1)*binsize):
					histogramcounter[j]= histogramcounter[j] + 1

	# due to bad coding I add the largest value manually (dumb but works)
	histogramcounter[bins-1] = histogramcounter[bins-1] + 1
	
	# generate X axis of the histogram
	histoX = []
	for i in range(0,bins):
		histoX.append(minarray + i*binsize + binsize/2.0 )

	# normalization
	histoNORM = []
	norm = 0.0
	for i in range(0,bins):
		histoNORM.append((bins/totalsize)*(histogramcounter[i]/len(array)))
		norm = norm + (bins/totalsize)*(histogramcounter[i]/len(array))*binsize
	
	return histogramcounter,binsize, histoX, histoNORM			# OUTPUT: unnormed histogram, bins size, histogram X azis, normed histogram Y axis


def Maximally_entangled_Bell_state(N: int, theta: float):

    zero = np.array([0.0,1.0])
    one  = np.array([1.0,0.0])


    zero_iterate = zero
    one_iterate  = one
    for i in range(0,N-1):
        zero_iterate = np.kron(zero_iterate,zero)
        one_iterate  = np.kron(one_iterate, one)

    psi = np.cos(theta)*zero_iterate  + np.sin(theta)*one_iterate

    return psi

def vonNeumann_test(N,R,psi):
    """ compute the von Neumann entropy of a state psi with the subsystem being size R  """

    # fix the R 
    R = int(math.floor(R))
    #R = int(math.ceil(R))
   
    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)

    ee = []
    for i in range(0,len(list_indic)):
        # compute the partial trace
        rho_reduced = PartialTraceGeneralTensor(N,list_indic[i],psi) 

        # compute eigenvalues
        eigenvalues,eigenvectors = LA.eigh(rho_reduced)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]

        # compute the von Neumann entropy
        entropy = 0.0
        epsi = 10**(-15)

        print('hahaa = ',len(eigenvalues))

        for ii in range(0,len(eigenvalues)):
            if (eigenvalues[ii] > epsi):
                entropy += eigenvalues[ii]*np.log2(eigenvalues[ii])
               # print(ii,eigenvalues[ii])

#        print("entropy------------------------------------------")
 #       print(entropy)
        ee.append(entropy)        


    entropy = - np.average(ee)


    return entropy,ee



def ParityOperator(N):

    sX, sY, sZ, sI = PauliMatrice()

    tampon = sZ

    for i in range(0,N-1):
        tampon = np.kron(sZ,tampon)

    return tampon

def ParityOperatorGeneral(N: int, which: int):
    """

    Compute and output different parity operators

    Arguments:
        N (integer): what is the number of spins in the 2**N space
        which (integer): which out of the 4 Paulies we build the parity operator out of    
    Returns:
        tampon(ndarray matrix): the parity operator matrix

    """

    sX, sY, sZ, sI = PauliMatrice()

    if (which == 0):
        kk = sI 
    elif(which == 1):
        kk = sX 
    elif(which == 2):
        kk = sY
    elif(which == 3):
        kk = sZ
    else:
        return print("ERROR")
    tampon = kk

    for i in range(0,N-1):
        tampon = np.kron(kk,tampon)

    return tampon





def FullHamiltonianBuild_XYZ(Jx: float,Jy: float,Jz: float,h: float, usePBC: bool, size: int):

    sX, sY, sZ, sI = PauliMatrice()

    
    hamiltonian = np.zeros([2**(size),2**(size)])

    for i in range(1,size):
        hamiltonian = hamiltonian + Jx*(FullSpace(sX,i,size) @ FullSpace(sX,i+1,size)) + Jy*(FullSpace(sY,i,size) @ FullSpace(sY,i+1,size)) + Jz*(FullSpace(sZ,i,size) @ FullSpace(sZ,i+1,size)) - h*(FullSpace(sZ,i,size))


    if usePBC:
        hamiltonian = hamiltonian + Jx*(FullSpace(sX,size,size) @ FullSpace(sX,1,size)) + Jy*(FullSpace(sY,size,size) @ FullSpace(sY,1,size)) + Jz*(FullSpace(sZ,size,size) @ FullSpace(sZ,1,size)) - h*(FullSpace(sZ,size,size))


    return hamiltonian


def FullSpace(operator: np.ndarray, position: int, size: int):

    sX, sY, sZ, sI = PauliMatrice()

    if (position == size):
        matrix = operator
    else:
        matrix = sI

    for i in range(size-1,0,-1):
        if (i == position):
            ope = operator
        else:
            ope = sI

        matrix = np.kron(ope,matrix)


    return matrix



## SPARSE DEFINTIONS OF THE PAULI OPERATORS
sx = sparse.csr_matrix(np.array([[0.,1.],[1.,0.]]),dtype=float)
sy = sparse.csr_matrix(np.array([[0.,-1.j],[1.j,0.]]))
sz = sparse.csr_matrix(np.array([[1.,0.],[0.,-1.]]),dtype=float)


class PauliOps:
    # this function comes from David Luitz exercise course at Les Houces
    def __init__(me,L):
        me.L = L
        me.X = []
        me.Y = []
        me.Z = []
        for i in range(L):
            leftdim=2**i
            rightdim=2**(L+1-i-2)
            x = sparse.kron(sparse.kron(sparse.eye(leftdim),sx),sparse.eye(rightdim),'csr') 
            y = sparse.kron(sparse.kron(sparse.eye(leftdim),sy),sparse.eye(rightdim),'csr')
            z = sparse.kron(sparse.kron(sparse.eye(leftdim),sz),sparse.eye(rightdim),'csr')
            me.X.append(x)
            me.Y.append(y)
            me.Z.append(z)

            

class XYZ_Hamiltonian:
    # this function comes from David Luitz exercise course at Les Houces
    def __init__(me,L,Jx,Jy,Jz,h,pbc): 
        me.L=L
        me.H = sparse.csr_matrix((2**L,2**L))  
        paulis = PauliOps(L)

        me.Jx = Jx
        me.Jy = Jy
        me.Jz = Jz
        me.h  = h
        
        
        if pbc:
            Lb=L
            me.bc = "periodic boundaries"
        else:
            Lb=L-1
            me.bc = "open boundaries"
        
        
        for i in range(L):
            me.H += h*paulis.Z[i]
            

        for i in range(Lb):
            me.H += Jx*(paulis.X[i]*paulis.X[(i+1)%L]) + Jy*(paulis.Y[i]*paulis.Y[(i+1)%L]) + Jz*(paulis.Z[i]*paulis.Z[(i+1)%L])



def compute_magicSRE(state: np.ndarray, N: int):

    """
     Compute the magic measure according to the definition by  Leone, L., Oliviero, S. F. E. & Hamma, A. Stabilizer Renyi Entropy. Phys. Rev. Lett. 128, 050402 (2022).

    Arguments:
        state(numpy array, vector of size 2^N): state as input from which we compute the magic
    Returns:
        value_numerical(float): SRE magic value

    """

    alpha = 2.0
    sX, sY, sZ, sI = PauliMatrice()


    # basic computations
    gate_list = [sI,sX,sY,sZ]
    gate_list_id = [0,1,2,3]

    temp = [p for p in itertools.product(gate_list_id, repeat=N)]
    kk = 0.0
    tampon = 0.0
    for i in range(0,len(temp)):

        gate = gate_list[temp[i][0]]
        for j in range(1,len(temp[i])):
            gate = np.kron(gate,gate_list[temp[i][j]])


        theta =  (2.0**(-N))*(np.conjugate(np.transpose(state)) @ gate @ state)**(2.0)

        kk = kk + theta

        tampon = tampon + theta**alpha

    value_numerical = (1.0/(1.0-alpha))*np.log2(tampon) - N*np.log2(2.0)

    return value_numerical.real





def compute_magicSRE_list(state: np.ndarray ,N: int):

    """
     Compute the magic measure according to the definition by  Leone, L., Oliviero, S. F. E. & Hamma, A. Stabilizer Renyi Entropy. Phys. Rev. Lett. 128, 050402 (2022).

    Arguments:
        state(numpy array, vector of size 2^N): state as input from which we compute the magic
    Returns:
        value_numerical(float): SRE magic value

    """

    alpha = 2.0
    sX, sY, sZ, sI = PauliMatrice()


    # basic computations
    gate_list = [sI,sX,sY,sZ]
    gate_list_id = [0,1,2,3]

    temp = [p for p in itertools.product(gate_list_id, repeat=N)]
    

    kk = 0.0
    tampon = 0.0
    lista = []
    countx_list = []
    county_list = []
    countz_list = []
    string_list = []
    for i in range(0,len(temp)):

        gate = gate_list[temp[i][0]]
        for j in range(1,len(temp[i])):
            gate = np.kron(gate,gate_list[temp[i][j]])



        theta =  (np.conjugate(np.transpose(state)) @ gate @ state)

        kk = kk + theta

        countx = np.count_nonzero(np.array(temp[i]) == 1)
        county = np.count_nonzero(np.array(temp[i]) == 2)

        if (countx % 2 == 1) and (county % 2 == 1):
            # both are odd
            countz = 1
        else: 
            # both are even
            countz = 0


        countx_list.append(countx)
        county_list.append(county)
        countz_list.append(countz)
        string_list.append( temp[i] )

        print(i + 1,temp[i], round(theta.real,4), countx, county, countz)


        tampon = tampon + theta**alpha
        lista.append(theta.real)

        #print(i, kk, )

    value_numerical = (1.0/(1.0-alpha))*np.log2(tampon) - N*np.log2(2.0)

    return value_numerical.real, lista, string_list, countx_list, county_list, countz_list




def matrixexponential(input_matrix):
    """

    Matrix exponentiation

    Arguments:
        input_matrix (ndarray): matrix in the exponential
    Returns:
        output_matrix(ndarray matrix): output of the matrix exponentiationh

    """


    evals, right_evects     = LA.eig(input_matrix)
    output_matrix           = right_evects @ np.diag(np.exp(evals)) @ np.transpose(np.conjugate(right_evects))

  
    return output_matrix


def RandomQubitsRotation(N: int):
    """

    Generate random local rotation on all qubits

    Arguments:
        N (integer): what is the number of spins in the 2**N space
    Returns:
        tampon(ndarray matrix): rotation matrix on all qubits

    """

    sX, sY, sZ, sI = PauliMatrice()


    # choose the rotation paulies
    jedan   = sX
    dva     = sY
    tri     = sZ 


    theta =   np.random.uniform(0.0,2.0*np.pi,3) #np.zeros(3) + np.pi/4.0 #
    #theta = [theta[0],theta[1],theta[2]]
    #theta = [0.0,theta[1],0.0]



    tampon = matrixexponential(-1j*(theta[0]/2.0)*jedan) @ matrixexponential(-1j*(theta[1]/2.0)*dva) @ matrixexponential(-1j*(theta[2]/2.0)*tri)

    for i in range(0,N-1):

        theta =   np.random.uniform(0.0,2.0*np.pi,3) # np.zeros(3) + np.pi/4.0 #
        #theta = [0.0,theta[1],0.0]

        tampon = np.kron(matrixexponential(-1j*(theta[0]/2.0)*jedan) @ matrixexponential(-1j*(theta[1]/2.0)*dva) @ matrixexponential(-1j*(theta[2]/2.0)*tri),tampon)

    return tampon

def compute_magicSRE_basisrotation(state: np.ndarray ,N: int):

    """
     Compute the magic measure according to the definition by  Leone, L., Oliviero, S. F. E. & Hamma, A. Stabilizer Renyi Entropy. Phys. Rev. Lett. 128, 050402 (2022).

     BUT! we rotate the basis with random single qubit rotation on each spin

    Arguments:
        state(numpy array, vector of size 2^N): state as input from which we compute the magic
    Returns:
        value_numerical(float): SRE magic value

    """

    alpha = 2.0
    sX, sY, sZ, sI = PauliMatrice()

    state = RandomQubitsRotation(N) @ state

    # basic computations
    gate_list = [sI,sX,sY,sZ]
    gate_list_id = [0,1,2,3]

    temp = [p for p in itertools.product(gate_list_id, repeat=N)]
    kk = 0.0
    tampon = 0.0
    for i in range(0,len(temp)):

        gate = gate_list[temp[i][0]]
        for j in range(1,len(temp[i])):
            gate = np.kron(gate,gate_list[temp[i][j]])



        theta =  (2.0**(-N))*(np.conjugate(np.transpose(state)) @ gate @ state)**(2.0)

        kk = kk + theta

        tampon = tampon + theta**alpha
       # print(i,kk)

    value_numerical = (1.0/(1.0-alpha))*np.log2(tampon) - N*np.log2(2.0)

    return value_numerical.real




def AnalyticFiniteMomentum(N: int,p: float):
    """
    Simple evaluation of the analytic formula for magic of kink states with finite momentum

    Arguments:
        N (integer):     size of the finite system
        p (float):       momentum value
    Returns:
        value (float):

    """


    numerator   = 11.0 - 12.0*N + (np.sin((2.0 - 4.0*N)*p)/np.sin(2.0*p))
    denominator = 2.0*(N**(3.0))

    value = - np.log2(- numerator/denominator) 

    return value


def AnalyticZeroMomentum(N: int):
    """
    Simple evaluation of the analytic formula for magic of kink states with zero momentum

    Arguments:
        N (integer):     size of the finite system
    Returns:
        value (float):

    """

    value = 3.0*np.log2(N) - np.log2(7.0*N - 6.0) 

    return value


def AnalyticDIFFERENCE(N: int):
    """
    Analytic formula for the difference btween finite momentum and zero momentum SRE expressions. 

    Arguments:
        N (integer):     size of the finite system
    Returns:
        value (float):

    """

    value = np.log2((7.0*N - 6.0)/(6*N - 6.0))

    return value




def DiagonalizeAndSelectXYZ(N: int,Jx: float,Jy: float,Jz: float,h: float,usePBC: bool):
    """

    Function that picks out the ground states of a degenerate spin model with frustration with positive and negative momentum

    After the projection and translation procedure is done a single state is produced using which the SRE magic is computed.


    Arguments:
        N (integer):     system size/ number of spins
        Jx (float):      along x axis coupling between neighboring spins
        Jy (float):      along y axis coupling between neighboring spins
        Jz (float):      along z axis coupling between neighboring spins
        h (float):       global magnetic field long z axis
        usePBC(boolean)  if we should use periodic boundary conditions or not
    Returns:
        state (2d numpy array ndarray): state vector
        momentum (float):               finite momentum associated with the state vector

    """



    ########## define the full sparse Hamiltonian (we have verified that using the sparse produces correctly everything as it should with other codes)
    HAMILTONIAN = XYZ_Hamiltonian(N,Jx,Jy,Jz,h,usePBC)
    H = HAMILTONIAN.H  # H is a sparse hermitian matrix


    # parity operators projections
    parityoperator = ParityOperator(N)
    Pze = (1.0/2.0)*(np.eye(2**(N)) + parityoperator)
    Pzo = (1.0/2.0)*(np.eye(2**(N)) - parityoperator)
    sparse_Pze = sparse.csr_matrix(Pze)
    sparse_Pzo = sparse.csr_matrix(Pzo)

    # generating the projected Hamiltonian on even and odd sector
    Hame = sparse_Pze @ H @ sparse_Pze;                                                            
    Hamo = sparse_Pzo @ H @ sparse_Pzo;

    # solve the eigenvalue problems for the respective sectors
    En_even, Vn_even = np.linalg.eigh(Hame.toarray()) # sparse solver 
    min_even = min(En_even)

    En_odd, Vn_odd = np.linalg.eigh(Hamo.toarray()) # sparse solver
    min_odd = min(En_odd)

    # figure out in which sector the ground-state minimum is
    E0 = min_even

    # define the translation operator (section 6.1.3 in Sutherland, B. Beautiful Models: 70 Years of Exactly Solved Quantum Many-Body Problems. (World Scientific Publishing Company, 2004).) and turn into a sparse matrix
    T = XYZchain_LatticeTranslationOperator(N)
    sparse_T = sparse.csr_matrix(T)

    # translate the projected Hamiltonians on the even or odd sector depending which admits the ground-state energy minimum
    hh = sparse_T @ Hame @ sparse_T 

    if (min_odd < E0):
        E0 = min_odd

        hh = sparse_T @ Hamo @ sparse_T    

    dense_hh = hh.toarray()

    # we now have figured-out what is the corrected sector (odd or even) that needs to be considered, and we have it translated using the translation operator
    eigenvalues, eigenvectors = np.linalg.eig(dense_hh)

    # Sort the eigenvalues by magnitude
    sorted_indices = np.argsort(np.abs(eigenvalues))

    # Sort the eigenvectors using the sorted indices
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]



    # we now sort-out the two degenerate states one with negative momentum one positive
    # here was a problem in the code for specific cases so instead of adding two degenerate states I expanded to 10, so i case more heave degeneracy happens we can still write in the array and take the one we
    gs = np.zeros([10,2**(N)],dtype=complex)
    counter = 0

    for i in range(0,len(eigenvalues)):
        
        arg = math.atan2((eigenvalues[i]/E0).imag, (eigenvalues[i]/E0).real)/(2.0*np.pi)

        print('lol', counter,i, eigenvalues[i],abs(eigenvalues[i]),E0)


        # the eigenvalue magnitude has to be the same before and after translating with operator T
        if (abs(abs(E0) - abs(eigenvalues[i])) < 1E-10):
            gs[counter,:] = eigenvectors[:,i].copy()
            counter = counter + 1
            momentum = arg
            #print('lol', counter,i, eigenvalues[i],abs(eigenvalues[i]))
    

    #print('degeneracy =', eigenvalues)
    # pick out one of the states
    state = gs[0,:].copy()

    return state, momentum, E0







def generate_kink_finitemomentum(N: int,mode: int,p: float):
    """

   Function to generate kink state in full 2**N dimensionality with finite momentum. 

    Arguments:
        N (integer):     system size/ number of spins
        mode (integer):  do we consider PBC (value 1) or OBC (value 0)
        p (float):       momentum value
    Returns:
        state (ndarray): state output
        

    """


    kink_list, kink_int_list = kink_state_generator_AF(N,mode,0)
    #print(kink_list)

    kink_list_size = len(kink_list)
    zero    = np.array([1.0,0.0])
    one     = np.array([0.0,1.0])

    # generate the total state
    totstate = np.zeros(int(2**N))

    for jj in range(0,kink_list_size):
        
        # last element
        if (int(kink_list[jj][N-1]) == 0):
            last = zero.copy()
        elif(int(kink_list[jj][N-1]) == 1):
            last = one.copy()
        else:
            print('ERROR')  

        for j in range(2,N+1):

            if (int(kink_list[jj][N-j]) == 0):
                tampon = zero.copy()
            elif(int(kink_list[jj][N-j]) == 1):
                tampon = one.copy()
            else:
                print('ERROR')  

            last = np.kron(tampon, last)


        totstate = totstate + (1.0/np.sqrt(2.0*N))*np.exp(1j*(jj+1)*p)*last

    kink_list, kink_int_list = kink_state_generator_AF(N,mode,1)
    #print(kink_list)

    for jj in range(0,kink_list_size):
        
        # last element
        if (int(kink_list[jj][N-1]) == 0):
            last = zero.copy()
        elif(int(kink_list[jj][N-1]) == 1):
            last = one.copy()
        else:
            print('ERROR')  

        for j in range(2,N+1):

            if (int(kink_list[jj][N-j]) == 0):
                tampon = zero.copy()
            elif(int(kink_list[jj][N-j]) == 1):
                tampon = one.copy()
            else:
                print('ERROR')  

            last = np.kron(tampon, last)


        totstate = totstate + (1.0/np.sqrt(2.0*N))*np.exp(-1j*(jj+1)*p)*last


    return totstate

def generate_kink_finitemomentum2(N: int,mode: int,p: float):
    """

   Function to generate kink state in full 2**N dimensionality with finite momentum. 

   THIS IS MODIFIED SO THE MOMENTUM IS THE SAME BETWEEN THE TWO SPECIES PARITY KINK STATES

    Arguments:
        N (integer):     system size/ number of spins
        mode (integer):  do we consider PBC (value 1) or OBC (value 0)
        p (float):       momentum value
    Returns:
        state (ndarray): state output
        

    """


    kink_list, kink_int_list = kink_state_generator_AF(N,mode,0)
    #print(kink_list)

    kink_list_size = len(kink_list)
    zero    = np.array([1.0,0.0])
    one     = np.array([0.0,1.0])

    # generate the total state
    totstate = np.zeros(int(2**N))

    for jj in range(0,kink_list_size):
        
        # last element
        if (int(kink_list[jj][N-1]) == 0):
            last = zero.copy()
        elif(int(kink_list[jj][N-1]) == 1):
            last = one.copy()
        else:
            print('ERROR')  

        for j in range(2,N+1):

            if (int(kink_list[jj][N-j]) == 0):
                tampon = zero.copy()
            elif(int(kink_list[jj][N-j]) == 1):
                tampon = one.copy()
            else:
                print('ERROR')  

            last = np.kron(tampon, last)


        totstate = totstate + (1.0/np.sqrt(2.0*N))*np.exp(1j*(jj+1)*p)*last

    kink_list, kink_int_list = kink_state_generator_AF(N,mode,1)
    #print(kink_list)

    for jj in range(0,kink_list_size):
        
        # last element
        if (int(kink_list[jj][N-1]) == 0):
            last = zero.copy()
        elif(int(kink_list[jj][N-1]) == 1):
            last = one.copy()
        else:
            print('ERROR')  

        for j in range(2,N+1):

            if (int(kink_list[jj][N-j]) == 0):
                tampon = zero.copy()
            elif(int(kink_list[jj][N-j]) == 1):
                tampon = one.copy()
            else:
                print('ERROR')  

            last = np.kron(tampon, last)


        totstate = totstate + (1.0/np.sqrt(2.0*N))*np.exp(1j*(jj+1)*p)*last


    return totstate



def kink_state_generator_AF(N: int,mode: int,which: int):
    """

   Function to generate the kink state lists of numbers

    Arguments:
        N (integer):     system size/ number of spins
        mode (integer):  do we consider PBC (value 1) or OBC (value 0)
        which (integer): which parity kink states we want, so which of the two Neel order states with the embedded corresponding kink we want (possiblities are 0 or 1) 
    Returns:
        state_list (string array or list of strings): returns the list of kink states of fixed parity
        integer_list (integer array): spits out the list of integer ids of the kink states in the whole possible Hilbert state space 

    """


    # generate the list to fill out
    integer_list = []
    state_list = []

    # generate the first kink state in PBC settings 
    first_state =  ''
    for i in range(0,N):
        first_state = first_state + (str(i % 2))

    # generate the second (the same as first but globally inverted) kink state in PBC settings
    second_state = ''
    for i in range(0,N):
        second_state = second_state + (str(((i + 1) % 2) ))

    # loop overt the first state to generate all the possible kink states
    if (which == 0):
        loop_state = first_state
    if (which == 1):
        loop_state = second_state


    for j in range(0,N-1):
        new_state = str(loop_state[N-1])
        for i in range(1,N):
            new_state = new_state + str(loop_state[i-1])
        state_list.append(new_state)
        integer_list.append(int(new_state,2))

        loop_state = new_state


    if (mode == 1):
        if (which == 0):
            integer_list.append(int(first_state,2))     
            state_list.append(first_state)
        if (which == 1):
            integer_list.append(int(second_state,2))     
            state_list.append(second_state)


    return state_list,integer_list


def Renyi2DIRECT(N,A,psi):
    """

    compute renyi2 for just the thing we are looking for

    """

    rho_reduced = PartialTraceGeneralTensor(N,A,psi) 

    entropy = -np.log((rho_reduced * rho_reduced.T).sum())                              #-np.log(np.trace(rho_reduced @ rho_reduced)) #-np.log(np.sum(np.square(rho_reduced)))

    return entropy





"""

Analytic formulas for the von Neumann entropy and general Renyi entropy with parameter 

"""
def RSA(r,alpha):
    return (1.0/(1.0 - alpha))*np.log(r**alpha + (1.0 - r)**alpha)

def RSB(r,l,alpha):
    return (1.0/(1.0 - alpha))*np.log(2.0*(r/2)**alpha + l**alpha +  (1.0 - r - l)**alpha)    

def RSAUB(r,l,alpha):
    return (1.0/(1.0 - alpha))*np.log(r**alpha + l**alpha + (r/2)**alpha + (1 - l - (3.0/2.0)*r)**alpha)

def RSAIB(r,alpha):
    return (1.0/(1.0 - alpha))*np.log((r/2.0)**alpha + (1.0 - (r/2.0))**alpha)

def SA(r):
    return -r*np.log2(r) - (1.0 - r)*np.log2(1.0 - r)

def SB(r,l):
    return -r*np.log2(r/2.0) - l*np.log2(l) -  (1.0 - r - l)*np.log2(1.0 - r - l)

def SAUB(r,l):
    return -r*np.log2(r) - l*np.log2(l) - (r/2.0)*np.log2(r/2.0) - (1.0 - l - (3.0/2.0)*r)*np.log2((1.0 - l - (3.0/2.0)*r))

def SAIB(r):
    return -(r/2.0)*np.log2(r/2.0) - (1.0 - (r/2.0))*np.log2(1.0 - (r/2.0))


"""
Given the size and subsystem compute the eignvalues of the reduced density matrix. 

"""
def compute_everything(A,N):
    """
    
    Provided with a partition and system size use the kink state numerics to obtain the reduced density matrix and the entanglement entropy

    Arguments:
        A (int array):   array with the subsystem spins
        N (integer):     system size/ number of spins
    Returns:
        

    """    

    mode = 1 # 1 for PBC, 0 for OBC

    kink_list, kink_int_list,rho_reduced_new, basis = kink_state_generator_AF_ALL(N, mode, A)
    kink_list_size = len(kink_list)

    eigenvalues,eigenvectors = LA.eigh(rho_reduced_new)
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]

    epsi = 10**(-15)


    # compute the von Neumann entropy
    entropy = 0.0
    for ii in range(0,len(eigenvalues)):
        if (eigenvalues[ii] > epsi):
            entropy += -eigenvalues[ii]*np.log2(eigenvalues[ii])

    return entropy






def compute_everythingXY(A,N):
    """
    
    Provided with a partition and system size use the kink state numerics to obtain the reduced density matrix and the entanglement entropy

    Arguments:
        A (int array):   array with the subsystem spins
        N (integer):     system size/ number of spins
    Returns:
        

    """    

    mode = 1 # 1 for PBC, 0 for OBC

    kink_list, kink_int_list,rho_reduced_new, basis = kink_state_generator_AF_XYmodel(N, mode, A)
    kink_list_size = len(kink_list)

    eigenvalues,eigenvectors = LA.eigh(rho_reduced_new)
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]

    epsi = 10**(-15)


    # compute the von Neumann entropy
    entropy = 0.0
    for ii in range(0,len(eigenvalues)):
        if (eigenvalues[ii] > epsi):
            entropy += -eigenvalues[ii]*np.log2(eigenvalues[ii])

    return entropy



"""
Function that generates the list of possible kink state (AntiFerroMagnetic) manually by iterating over the list

Input: 
- number of spins N
- mode: 1 for PBC, 0 for OBC

Output: list of 2*N kink states that live in the 2^N Hilbert space

"""
def kink_state_generator_AF_ALL(N,mode,index_list):

    # generate the list to fill out
    integer_list = []
    state_list = []

    # generate the first kink state in PBC settings 
    first_state =  ''
    for i in range(0,N):
        first_state = first_state + (str(i % 2))

    # generate the second (the same as first but globally inverted) kink state in PBC settings
    second_state = ''
    for i in range(0,N):
        second_state = second_state + (str(((i + 1) % 2) ))

    # loop overt the first state to generate all the possible kink states
    loop_state = first_state
    for j in range(0,N-1):
        new_state = str(loop_state[N-1])
        for i in range(1,N):
            new_state = new_state + str(loop_state[i-1])
        state_list.append(new_state)
        integer_list.append(int(new_state,2))

        loop_state = new_state


    if (mode == 1):
        integer_list.append(int(first_state,2))     
        state_list.append(first_state)


    # figure out what is the subsystem spins
    index_list = np.array(index_list) - 1
    trace_out_indices = np.delete(np.arange(N), index_list)
    #print('sub domains = ', index_list,trace_out_indices)

    tampon_traced_state = []
    tampon_domain_state = []

    for i in range(0,len(state_list)):
        tampon_state = state_list[i]
        stat = ''
        for j in range(0,len(trace_out_indices)):
            index = trace_out_indices[j]
            stat = stat + str(tampon_state[index])
        tampon_traced_state.append(stat)

        stat = ''
        for j in range(0,len(index_list)):
            index = index_list[j]
            stat = stat + str(tampon_state[index])


        tampon_domain_state.append(stat)

    #print('sub domain spins = ', tampon_domain_state,tampon_traced_state)

    D = defaultdict(list)
    for i,item in enumerate(tampon_domain_state):
        D[item].append(i)
    D = {k:v for k,v in D.items() if len(v)>1}

    #print('BigD =', D)

    a = dict(Counter(tampon_domain_state))
    #print('zzz - ', a)
    basis = []
    diag = []
    for k,v in a.items():
        basis.append(k)
        diag.append(v)


    rho_reduced_new = np.zeros((len(diag),len(diag)))
    for i in range(0,len(diag)):
        rho_reduced_new[i,i] = diag[i]/N

    ttx = []
    tty = []
    for i in range(0,len(tampon_traced_state)):
        for j in range(0,len(tampon_traced_state)):
            if (i < j):
                if (tampon_traced_state[i] == tampon_traced_state[j]):
                    ttx.append(i)
                    tty.append(j)


    for i in range(0,len(ttx)):
        #print(i,ttx[i],tty[i],tampon_domain_state[ttx[i]])
        for j in range(0,len(basis)):
            if (tampon_domain_state[ttx[i]] == basis[j]):
                first_index = j

        for j in range(0,len(basis)):
            if (tampon_domain_state[tty[i]] == basis[j]):
                second_index = j

        rho_reduced_new[first_index,second_index] = rho_reduced_new[first_index,second_index] + 1.0/N
        rho_reduced_new[second_index,first_index] = rho_reduced_new[second_index,first_index] + 1.0/N


    return state_list, integer_list,rho_reduced_new, basis



"""
Function that generates the list of possible kink state (AntiFerroMagnetic) manually by iterating over the list

Input: 
- number of spins N
- mode: 1 for PBC, 0 for OBC

Output: list of 2*N kink states that live in the 2^N Hilbert space

"""
def kink_state_generator_AF_XYmodel(N,mode,index_list):

    # generate the list to fill out
    integer_list = []
    state_list = []

    # generate the first kink state in PBC settings 
    first_state =  ''
    for i in range(0,N):
        first_state = first_state + (str(i % 2))

    # generate the second (the same as first but globally inverted) kink state in PBC settings
    second_state = ''
    for i in range(0,N):
        second_state = second_state + (str(((i + 1) % 2) ))

    # loop overt the first state to generate all the possible kink states
    loop_state = first_state
    for j in range(0,N-1):
        new_state = str(loop_state[N-1])
        for i in range(1,N):
            new_state = new_state + str(loop_state[i-1])
        state_list.append(new_state)
        integer_list.append(int(new_state,2))

        loop_state = new_state


    if (mode == 1):
        integer_list.append(int(first_state,2))     
        state_list.append(first_state)


    # figure out what is the subsystem spins
    index_list = np.array(index_list) - 1
    trace_out_indices = np.delete(np.arange(N), index_list)
    #print('sub domains = ', index_list,trace_out_indices)

    tampon_traced_state = []
    tampon_domain_state = []

    for i in range(0,len(state_list)):
        tampon_state = state_list[i]
        stat = ''
        for j in range(0,len(trace_out_indices)):
            index = trace_out_indices[j]
            stat = stat + str(tampon_state[index])
        tampon_traced_state.append(stat)

        stat = ''
        for j in range(0,len(index_list)):
            index = index_list[j]
            stat = stat + str(tampon_state[index])


        tampon_domain_state.append(stat)

    #print('sub domain spins = ', tampon_domain_state,tampon_traced_state)

    # make a list of repeating spins in the domain
    D = defaultdict(list)
    for i,item in enumerate(tampon_domain_state):
        D[item].append(i)
    D = {k:v for k,v in D.items() if len(v)>1}

    #print('BigD =', D)

    # make a list of repeating traced out spins
    Dtraced = defaultdict(list)
    for i,item in enumerate(tampon_traced_state):
        Dtraced[item].append(i)
    Dtraced = {k:v for k,v in Dtraced.items() if len(v)>1}

    #print('BigDtraced =', Dtraced)

    # generate the prefactors from the traced spins
    Dkey = [*Dtraced]          #list of keys
    Dval = [*Dtraced.values()] #list of values

    #print('true index = ',Dkey,Dval, Dval[0][0],len(Dval))

    p = (np.pi/2.0)*(1.0 - 1.0/N)
    coef_list = []
    for i in range(0,len(Dval)):
        for ii in range(1,len(Dval[i])):
            Dval_first = Dval[i][0] + 1
            Dval_second = Dval[i][ii] + 1
        
            coef_list.append(np.exp(1j*p*(Dval_first - Dval_second)))
            #print('torrre = ', Dval_first,Dval_second)

    #print('coeff = ', coef_list)
    a = dict(Counter(tampon_domain_state))
    #print('zzz - ', a)
    basis = []
    diag = []
    for k,v in a.items():
        #print('basis = ',k,v)
        basis.append(k)
        diag.append(v)


    rho_reduced_new = np.zeros((len(diag),len(diag)),dtype=complex)
    for i in range(0,len(diag)):
        rho_reduced_new[i,i] = diag[i]/N

    ttx = []
    tty = []
    coef_list2 = []

    for i in range(0,len(tampon_traced_state)):
        for j in range(0,len(tampon_traced_state)):
            if (i < j):
                if (tampon_traced_state[i] == tampon_traced_state[j]):
                    #print('lol=', i,j,tampon_traced_state[i],tampon_traced_state[j])
                    ttx.append(i)
                    tty.append(j)
                    coef_list2.append(np.exp(1j*p*(i - j)))

    #print('hahaha =', coef_list2)
    for i in range(0,len(ttx)):
        #print(i,ttx[i],tty[i],tampon_domain_state[ttx[i]])
        for j in range(0,len(basis)):
            if (tampon_domain_state[ttx[i]] == basis[j]):
                first_index = j

        for j in range(0,len(basis)):
            if (tampon_domain_state[tty[i]] == basis[j]):
                second_index = j


        #print('jebeno =', i, first_index ,second_index, tampon_domain_state[ttx[i]],tampon_domain_state[tty[i]])
        rho_reduced_new[first_index,second_index] = rho_reduced_new[first_index,second_index] + coef_list2[i]/N
        rho_reduced_new[second_index,first_index] = rho_reduced_new[second_index,first_index] + np.conjugate(coef_list2[i])/N


    return state_list, integer_list,rho_reduced_new, basis




def kink_state_generator_W(N):

    # generate the list to fill out
    integer_list = []
    state_list = []

    # generate the first kink state in PBC settings 
    first_state =  ''
    for i in range(0,N):
        first_state = first_state + str(0)



    for i in range(0,N):
        loop_state = list(first_state)
        loop_state[i] = '1' 
        loop_state = ''.join(loop_state)
        state_list.append(loop_state)


    return state_list


def generate_kink_stateW(N):

    kink_list = kink_state_generator_W(N)
    kink_list_size = len(kink_list)
    
    zero    = np.array([1.0,0.0])
    one     = np.array([0.0,1.0])

    # generate the total state
    totstate = np.zeros(int(2**N))

    for jj in range(0,kink_list_size):
        
        # last element
        if (int(kink_list[jj][N-1]) == 0):
            last = zero.copy()
        elif(int(kink_list[jj][N-1]) == 1):
            last = one.copy()
        else:
            print('ERROR')  

        for j in range(2,N+1):

            if (int(kink_list[jj][N-j]) == 0):
                tampon = zero.copy()
            elif(int(kink_list[jj][N-j]) == 1):
                tampon = one.copy()
            else:
                print('ERROR')  

            last = np.kron(tampon, last)


        totstate = totstate + (1.0/np.sqrt(N))*last





    return totstate

def generate_kink_stateWp(N):

    kink_list = kink_state_generator_W(N)
    kink_list_size = len(kink_list)
    
    zero    = np.array([1.0,0.0])
    one     = np.array([0.0,1.0])

    p = (np.pi/2.0)*(1.0 - 1.0/N)

    # generate the total state
    totstate = np.zeros(int(2**N))

    for jj in range(0,kink_list_size):
        
        # last element
        if (int(kink_list[jj][N-1]) == 0):
            last = zero.copy()
        elif(int(kink_list[jj][N-1]) == 1):
            last = one.copy()
        else:
            print('ERROR')  

        for j in range(2,N+1):

            if (int(kink_list[jj][N-j]) == 0):
                tampon = zero.copy()
            elif(int(kink_list[jj][N-j]) == 1):
                tampon = one.copy()
            else:
                print('ERROR')  

            last = np.kron(tampon, last)


        totstate = totstate + (1.0/np.sqrt(N))*np.exp(1j*(jj+1)*p)*last



    return totstate



def partition_vonNeuman(Nsites, list, psi):

    rho_reduced = PartialTraceGeneralTensor(Nsites,list,psi) 

    eigenvalues,eigenvectors = LA.eigh(rho_reduced)
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]

    # compute the von Neumann entropy
    entropy = 0.0
    epsi = 10**(-15)

    for ii in range(0,len(eigenvalues)):
        if (eigenvalues[ii] > epsi):
            entropy += eigenvalues[ii]*np.log2(eigenvalues[ii])

    return -entropy

def partition_vonNeumanTEST(Nsites, rho_reduced):

    eigenvalues,eigenvectors = LA.eigh(rho_reduced)
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]

    # compute the von Neumann entropy
    entropy = 0.0
    epsi = 10**(-15)

    for ii in range(0,len(eigenvalues)):
        if (eigenvalues[ii] > epsi):
            entropy += eigenvalues[ii]*np.log2(eigenvalues[ii])

    return -entropy

def generate_kink_stateTOT(N,mode):

    kink_list, kink_int_list = kink_state_generator_AF(N,mode,0)
    kink_list_size = len(kink_list)
    zero    = np.array([1.0,0.0])
    one     = np.array([0.0,1.0])

    # generate the total state
    totstate = np.zeros(int(2**N))

    for jj in range(0,kink_list_size):
        
        # last element
        if (int(kink_list[jj][N-1]) == 0):
            last = zero.copy()
        elif(int(kink_list[jj][N-1]) == 1):
            last = one.copy()
        else:
            print('ERROR')  

        for j in range(2,N+1):

            if (int(kink_list[jj][N-j]) == 0):
                tampon = zero.copy()
            elif(int(kink_list[jj][N-j]) == 1):
                tampon = one.copy()
            else:
                print('ERROR')  

            last = np.kron(tampon, last)


        totstate = totstate + (1.0/np.sqrt(2.0*N))*last


    kink_list, kink_int_list = kink_state_generator_AF(N,mode,1)
    kink_list_size = len(kink_list)

    for jj in range(0,kink_list_size):
        
        # last element
        if (int(kink_list[jj][N-1]) == 0):
            last = zero.copy()
        elif(int(kink_list[jj][N-1]) == 1):
            last = one.copy()
        else:
            print('ERROR')  

        for j in range(2,N+1):

            if (int(kink_list[jj][N-j]) == 0):
                tampon = zero.copy()
            elif(int(kink_list[jj][N-j]) == 1):
                tampon = one.copy()
            else:
                print('ERROR')  

            last = np.kron(tampon, last)


        totstate = totstate + (1.0/np.sqrt(2.0*N))*last


    return totstate


def generate_kink_stateTOTp(N,mode):


    p = (np.pi/2.0)*(1.0 - 1.0/N)


    kink_list, kink_int_list = kink_state_generator_AF(N,mode,0)
    kink_list_size = len(kink_list)
    zero    = np.array([1.0,0.0])
    one     = np.array([0.0,1.0])

    # generate the total state
    totstate = np.zeros(int(2**N))

    for jj in range(0,kink_list_size):
        
        # last element
        if (int(kink_list[jj][N-1]) == 0):
            last = zero.copy()
        elif(int(kink_list[jj][N-1]) == 1):
            last = one.copy()
        else:
            print('ERROR')  

        for j in range(2,N+1):

            if (int(kink_list[jj][N-j]) == 0):
                tampon = zero.copy()
            elif(int(kink_list[jj][N-j]) == 1):
                tampon = one.copy()
            else:
                print('ERROR')  

            last = np.kron(tampon, last)


        totstate = totstate + (1.0/np.sqrt(2.0*N))*np.exp(1j*(jj+1)*p)*last


    kink_list, kink_int_list = kink_state_generator_AF(N,mode,1)
    kink_list_size = len(kink_list)

    for jj in range(0,kink_list_size):
        
        # last element
        if (int(kink_list[jj][N-1]) == 0):
            last = zero.copy()
        elif(int(kink_list[jj][N-1]) == 1):
            last = one.copy()
        else:
            print('ERROR')  

        for j in range(2,N+1):

            if (int(kink_list[jj][N-j]) == 0):
                tampon = zero.copy()
            elif(int(kink_list[jj][N-j]) == 1):
                tampon = one.copy()
            else:
                print('ERROR')  

            last = np.kron(tampon, last)


        totstate = totstate + (1.0/np.sqrt(2.0*N))*np.exp(-1j*(jj+1)*p)*last


    return totstate




def chirality(psi: np.ndarray,N: int):
    """
    
    Computing the chirality of a provided state

    Arguments:
        psi (ndarray):   the state we want to consider
        N (integer):     system size/ number of spins
    Returns:
        chiral(float):   averaged chirality along all the spins
        
    """

    sX, sY, sZ, sI = PauliMatrice()

    # define the list of integers
    int_list = []
    for i in range(1,N + 1):
        int_list.append(i)
    
    # define the length of the list
    list_length = len(int_list)

    # loop through the list in pairs of three with periodic boundary conditions
    chiral = 0.0
    for i in range(list_length):
        i_mod_3 = i % 3
        j = (i + 1) % list_length
        k = (i + 2) % list_length
        
        lista = [int_list[i], int_list[j], int_list[k]]

        lista = np.sort(lista)

        # chiral operator definition parts
        a1 = FullSpaceThreeConcurrent(sX, lista[0], sY, lista[1], sZ, lista[2], N)
        a2 = FullSpaceThreeConcurrent(sY, lista[0], sZ, lista[1], sX, lista[2], N)
        a3 = FullSpaceThreeConcurrent(sZ, lista[0], sX, lista[1], sY, lista[2], N)

        a4 = FullSpaceThreeConcurrent(sX, lista[0], sZ, lista[1], sY, lista[2], N)
        a5 = FullSpaceThreeConcurrent(sY, lista[0], sX, lista[1], sZ, lista[2], N)
        a6 = FullSpaceThreeConcurrent(sZ, lista[0], sY, lista[1], sX, lista[2], N)

        # evaluating the chirality
        value = (np.transpose(np.conjugate(psi)) @ (a1 + a2 + a3 - a4 - a5 - a6) @ psi).real

        chiral = chiral + np.abs(value)

    a1 = np.transpose(np.conjugate(psi)) @ a1 @ psi
    a2 = np.transpose(np.conjugate(psi)) @ a2 @ psi
    a3 = np.transpose(np.conjugate(psi)) @ a3 @ psi
    a4 = np.transpose(np.conjugate(psi)) @ a4 @ psi
    a5 = np.transpose(np.conjugate(psi)) @ a5 @ psi
    a6 = np.transpose(np.conjugate(psi)) @ a6 @ psi

    return chiral/N,a1,a2,a3,a4,a5,a6





def FullSpaceThreeConcurrent(operator1: np.ndarray, position1: int, operator2: np.ndarray, position2: int, operator3: np.ndarray, position3: int, size: int):

    sX, sY, sZ, sI = PauliMatrice()

    if(position1 == size):
        matrix = operator1
    elif(position2 == size):
        matrix = operator2
    elif(position3 == size):
        matrix = operator3        
    else:
        matrix = sI

    for i in range(size-1,0,-1):
        if(i == position1):
            ope = operator1
        elif(i == position2):
            ope = operator2
        elif(i == position3):
            ope = operator3        
        else:
            ope = sI


        matrix = np.kron(ope,matrix)


    return matrix



def kink_state_generator_AF_GENERALREALCOEFFICIENTS2(N: int,mode: int, index_list: np.ndarray, coeff: np.ndarray):
    """
    
    Function to compute the Reduced Density Matrix of any partition you need and general coeffcient numbers either real or complex


    Arguments:
        N (integer):                        system size
        mode (integer):                     either 0 or 1 (zero for OBC and 1 for PBC)
        index_list(array of integers):      so list of the spins you want to keep, for example [1,2,3] will keep spins 1,2,3 and trace over the others
        coeff (list of float)               list of coefficients in front of basis of kink states either real/complex or whatever
    Returns:
        rho_reduced_new (2d matrix):        The reduced density matrix
      

    """



    # generate the list to fill out
    integer_list = []
    state_list = []

    # generate the first kink state in PBC settings 
    first_state =  ''
    for i in range(0,N):
        first_state = first_state + (str(i % 2))

    # generate the second (the same as first but globally inverted) kink state in PBC settings
    second_state = ''
    for i in range(0,N):
        second_state = second_state + (str(((i + 1) % 2) ))

    # loop overt the first state to generate all the possible kink states
    loop_state = first_state
    for j in range(0,N-1):
        new_state = str(loop_state[N-1])
        for i in range(1,N):
            new_state = new_state + str(loop_state[i-1])
        state_list.append(new_state)
        integer_list.append(int(new_state,2))

        loop_state = new_state


    if (mode == 1):
        integer_list.append(int(first_state,2))     
        state_list.append(first_state)


    # SECOND SET

    # loop overt the first state to generate all the possible kink states
    loop_state = second_state
    for j in range(0,N-1):
        new_state = str(loop_state[N-1])
        for i in range(1,N):
            new_state = new_state + str(loop_state[i-1])
        state_list.append(new_state)
        integer_list.append(int(new_state,2))

        loop_state = new_state


    if (mode == 1):
        integer_list.append(int(second_state,2))     
        state_list.append(second_state)



    #print(integer_list,state_list)

    # REMAINING OF THE ALGORITHM!

    # figure out what is the subsystem spins
    index_list = np.array(index_list) - 1
    trace_out_indices = np.delete(np.arange(N), index_list)
    #print('sub domains = ', index_list,trace_out_indices)

    tampon_traced_state = []
    tampon_domain_state = []

    for i in range(0,len(state_list)):
        tampon_state = state_list[i]
        stat = ''
        for j in range(0,len(trace_out_indices)):
            index = trace_out_indices[j]
            stat = stat + str(tampon_state[index])
        tampon_traced_state.append(stat)

        stat = ''
        for j in range(0,len(index_list)):
            index = index_list[j]
            stat = stat + str(tampon_state[index])


        tampon_domain_state.append(stat)

    #print('sub domain spins = ', tampon_domain_state,tampon_traced_state)

    # make a list of repeating spins in the domain
    D = defaultdict(list)
    for i,item in enumerate(tampon_domain_state):
        D[item].append(i)
    D = {k:v for k,v in D.items() if len(v)>1}

    #print('BigD =', D)

    # make a list of repeating traced out spins
    Dtraced = defaultdict(list)
    for i,item in enumerate(tampon_traced_state):
        Dtraced[item].append(i)
    Dtraced = {k:v for k,v in Dtraced.items() if len(v)>1}

    #print('BigDtraced =', Dtraced)

    # generate the prefactors from the traced spins
    Dkey = [*D]               #list of keys
    Dval = [*D.values()]      #list of values


    a = dict(Counter(tampon_domain_state))
    basis = []
    diag = []
    for k,v in a.items():
        #print('basis = ',k,v)
        basis.append(k)
        diag.append(v)


    rho_reduced_new = np.zeros((len(diag),len(diag)),dtype=complex)

    #print(tampon_domain_state)
    for i in range(0,len(diag)):
        add = 0.0
        for j in range(0,len(tampon_domain_state)):
            if (basis[i] == tampon_domain_state[j]):
                add = add + (coeff[j]*np.conjugate(coeff[j]))
        rho_reduced_new[i,i] = add

    # OFFDIAGONAL STUFF
    ttx = []
    tty = []
    coef_list2 = []

    for i in range(0,len(tampon_traced_state)):
        for j in range(0,len(tampon_traced_state)):
            if (i < j):
                if (tampon_traced_state[i] == tampon_traced_state[j]):
                    ttx.append(i)
                    tty.append(j)
                    coef_list2.append(coeff[i]*np.conjugate(coeff[j]))

    for i in range(0,len(ttx)):
        for j in range(0,len(basis)):
            if (tampon_domain_state[ttx[i]] == basis[j]):
                first_index = j

        for j in range(0,len(basis)):
            if (tampon_domain_state[tty[i]] == basis[j]):
                second_index = j


        rho_reduced_new[first_index,second_index] = rho_reduced_new[first_index,second_index] + coef_list2[i]
        rho_reduced_new[second_index,first_index] = rho_reduced_new[second_index,first_index] + np.conjugate(coef_list2[i])


    return state_list, integer_list,rho_reduced_new, basis




def diagonalization(matrix: np.ndarray):

    """
    Full matrix diagonalization which orderers eigenvalues and correspondingly eigenvectors

    Args:
        matrix (float twodimensional array, matrix):    matrix we want to diagonlize

    Returns:
        eigenvalues (float array ):                     ordered eigenvalues
        eigenvectors (float twodimensional array):      ordered eigenvectors

    """


    eigenvalues, eigenvectors = LA.eigh(matrix)   # LA.eigh(matrix) 
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]

    return eigenvalues, eigenvectors



def compute_magicSRE_fromRDM(DM: np.ndarray, N: int):

    """
    Computer the SRE for alpha = 2 in the case we are provided a density matrix (DM) or any reduced density matrix

    Consistent with the output of compute_magicSRE function

    Arguments:
        DM(numpy array, vector of size 2^N times 2^N): density matrix
        N (intger): system or subsystem size
    Returns:
        value_numerical(float): returns the value of SRE


    """

    sX, sY, sZ, sI = PauliMatrice()


    # basic computations
    gate_list = [sI,sX,sY,sZ]
    gate_list_id = [0,1,2,3]

    temp = [p for p in itertools.product(gate_list_id, repeat=N)]

    tampon1 = 0.0
    tampon2 = 0.0
    for i in range(0,len(temp)):

        gate = gate_list[temp[i][0]]
        for j in range(1,len(temp[i])):
            gate = np.kron(gate,gate_list[temp[i][j]])


        theta =  np.trace(gate @ DM)
        #print(i)
        # condition that circumvents division by zero when theta = 0j, otherwise useless
        if (abs(theta) > 10**(-40)):
            tampon1 = tampon1 + (theta**(4.0))
            tampon2 = tampon2 + (theta**(2.0))

        else:
            tampon1 = tampon1 + 0.0
            tampon2 = tampon2 + 0.0

    print(tampon1,tampon2)
    value_numerical = - np.log2(tampon1/tampon2) 



    return value_numerical.real


def CliffordOrbitBeyondLocal(state: np.ndarray, N: int, Nl: int):
    """

    For a given input state (vector) produce an output state (vector) that went through a random unitary evolution
    using local gates. The random unitary gates are: Hadamard, S phase gate, and CNOT gate. From this set of gates 
    we choose randomly and apply to the state. We apply N-1 for each neghboring pair of spins.
    We repeat that Nl times.

    """


    for z in range(0,Nl):
        for i in range(0,N-1):

            gate,ran = CliffordOrbit_select_gate()

            i_index = random.randint(1, N) 
            j_index = random.randint(1, N) 

            state = ApplyLocalGateB(gate,[i_index,j_index],state,N,2)

    return state


def CliffordOrbit_select_gate():


    sX, sY, sZ, sI = PauliMatrice()


    ran = np.random.randint(6)

    if (ran == 0):
        gate = CNOT_gate(2,1,2)
    elif (ran == 1):
        gate = CNOT_gate(2,2,1)
    elif (ran == 2):
        gate = np.kron(sI,Phase_gate(np.pi/2.0))
    elif (ran == 3):
        gate = np.kron(Phase_gate(np.pi/2.0), sI)
    elif (ran == 4):
        gate = np.kron(sI, Hadamard_gate())
    elif (ran == 5):
        gate = np.kron(Hadamard_gate(),sI)
    else:
        print("ERROR!!!")



    return gate, ran


def Hadamard_gate():
    """

    Definition of the Hadamard gate.


    """


    matrix = np.zeros([2,2])

    matrix[0,0] = 1.0
    matrix[0,1] = 1.0
    matrix[1,0] = 1.0
    matrix[1,1] = -1.0

    matrix = (1.0/np.sqrt(2.0))*matrix

    return matrix


def Phase_gate(theta: float):
    """

    Definition of the genetal Phase gate.

    """

    matrix = np.zeros([2,2],dtype=complex)

    matrix[0,0] = 1.0
    matrix[0,1] = 0.0
    matrix[1,0] = 0.0
    matrix[1,1] = np.exp(1j*theta)


    return matrix




def CNOT_gate(N,i,j):


    # generate the whole basis
    base_states=np.array([numberToBase(i, 2,N) for i in range(2**N)],dtype=int)
    # copy the states because we use base_states as auxilary
    list1 = base_states.copy()

    # list through all the 2**N states and apply the CNOT gate rules and produce a new list
    list2 = []
    for k in range(0,2**N):


        if ( (base_states[k,i-1] == 0) and (base_states[k,j-1] == 0) ):
            pass
        
        elif ( (base_states[k,i-1] == 0) and (base_states[k,j-1] == 1) ):
            pass
        elif ( (base_states[k,i-1] == 1) and (base_states[k,j-1] == 0) ):
            base_states[k,i-1] = 1
            base_states[k,j-1] = 1

        elif ( (base_states[k,i-1] == 1) and (base_states[k,j-1] == 1) ):
            base_states[k,i-1] = 1
            base_states[k,j-1] = 0
        
        list2.append(base_states[k])

    # generate a matrix of zeros
    matrix = np.zeros([2**N,2**N])

    # now compare list1 and list2 which are the indices of the full dimensionality CNOT gate, only N are one, remaining ones are zero
    for k in range(0,len(list1)):
        kk1 = ''.join(map(str,list1[k]))
        kk2 = ''.join(map(str,list2[k]))

        matrix[int(kk1,2),int(kk2,2)] = 1
    

    return matrix

# auxiliry code to generate bit strings of size N
def numberToBase(n, b,n_qubits):
    if n == 0:
        return np.zeros(n_qubits,dtype=int)
    digits = np.zeros(n_qubits,dtype=int)
    counter=0
    while n:
        digits[counter]=int(n % b)
        n //= b
        counter+=1
    return digits[::-1]





def ApplyLocalGateB(sigma: np.ndarray, index_pair: np.ndarray, psi: np.ndarray, N: int, d: int):
    """ 
    Apply exponentiated local 2 site gate, this is generalized to any distance pair but does not involve pairs that are there due to PBC

    Arguments:
        sigma(float array): gate which we have to apply
        index_pair(integer array): list of indices where we apply the local gate
        psi(float array): the wave-function
        N(integer): number of spins
        d(integer): size of local dimension
    Returns:
        psi(float array): the wave-function after we applied the local gate


    """


    j = index_pair[0]


    # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi                           = psi.reshape(reshape_array_default)

    if (j == N):

        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default1       = np.arange(N)
        transpose_array_default2       = transpose_array_default1.copy()

        # special case
        if (N == 2):

            transpose_array_default1[0]    = 1
            transpose_array_default1[1]    = 0

            transpose_array_default1[0]    = 0
            transpose_array_default1[1]    = 1

        else:

            transpose_array_default1[0]    = N-1
            transpose_array_default1[1]    = 0
            transpose_array_default1[N-1]  = 1

            transpose_array_default2[0]    = 1
            transpose_array_default2[1]    = N-1
            transpose_array_default2[N-1]  = 0


        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)

    elif(j == 2):

        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default1       = np.arange(N)
        transpose_array_default2       = transpose_array_default1.copy()

        transpose_array_default1[0]    = 1
        transpose_array_default1[1]    = 2
        transpose_array_default1[2]    = 0

        transpose_array_default2[0]    = 2
        transpose_array_default2[1]    = 0
        transpose_array_default2[2]    = 1

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)
      


    else:    
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default       = np.arange(N)
        transpose_array_default[0]    = j-1
        transpose_array_default[1]    = j
        transpose_array_default[j-1]  = 0
        transpose_array_default[j]    = 1

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)


    return psi


def FourierMatrix(n: int):
    """

    As it is defined in Mathematica and wikipedia the discrete (quantum) Fourier Transform.

    """

    matrix = np.zeros([2**n,2**n],dtype=complex)

    prefactor = 1.0/(np.sqrt(2**n))

    for i in range(0,2**n):
        for j in range(0,2**n):

            matrix[i,j] = prefactor*np.exp((2.0*np.pi*1j*i*j)/(2**n) )

    return matrix



def U1N3():


    zerozeero = np.array([[1.0,0.0],[0.0,0.0]])
    oneone  = np.array([[0.0,0.0],[0.0,1.0]])

    matrix = np.kron(Hadamard_gate(),np.kron(np.eye(2),np.eye(2)))

    return matrix


def U2N3():

    zerozero = np.array([[1.0,0.0],[0.0,0.0]])
    oneone  = np.array([[0.0,0.0],[0.0,1.0]])

    matrix = np.kron(oneone,np.kron(Phase_gate(np.pi/2.0),np.eye(2)))
    matrix += np.kron(zerozero,np.kron(np.eye(2),np.eye(2)))

    return matrix


def U3N3():

    zerozero = np.array([[1.0,0.0],[0.0,0.0]])
    oneone  = np.array([[0.0,0.0],[0.0,1.0]])

    matrix = np.kron(oneone,np.kron(np.eye(2),Phase_gate(np.pi/4.0)))
    matrix += np.kron(zerozero,np.kron(np.eye(2),np.eye(2)))


    return matrix

def U4N3():

    zerozero = np.array([[1.0,0.0],[0.0,0.0]])
    oneone  = np.array([[0.0,0.0],[0.0,1.0]])

    matrix = np.kron(np.eye(2),np.kron(Hadamard_gate(),np.eye(2)))


    return matrix


def U5N3():

    zerozero = np.array([[1.0,0.0],[0.0,0.0]])
    oneone  = np.array([[0.0,0.0],[0.0,1.0]])

    matrix = np.kron(np.eye(2),np.kron(oneone,Phase_gate(np.pi/2.0)))
    matrix += np.kron(np.eye(2),np.kron(zerozero,np.eye(2)))


    return matrix

def U6N3():

    zerozero = np.array([[1.0,0.0],[0.0,0.0]])
    oneone  = np.array([[0.0,0.0],[0.0,1.0]])

    matrix = np.kron(np.eye(2),np.kron(np.eye(2),Hadamard_gate()))

    return matrix

def U7N3():

    zerozero = np.array([[1.0,0.0],[0.0,0.0]])
    oneone  = np.array([[0.0,0.0],[0.0,1.0]])


    sX, sY, sZ, sI = PauliMatrice()


    matrix = 0.5*np.kron(np.eye(2,dtype=complex),np.kron(np.eye(2,dtype=complex),np.eye(2,dtype=complex)))
    matrix += 0.5*np.kron(sX,np.kron(np.eye(2),sX))
    matrix += 0.5*np.kron(sY,np.kron(np.eye(2),sY))
    matrix += 0.5*np.kron(sZ,np.kron(np.eye(2),sZ))


    return matrix


def MlinBound(L: int):


    value = 1.0 - (1.0/(4.0 + 2**L))*(4.0 + ((2**L + 3)/(4.0))*(2**(-L*np.log(L))))

    return value


# Define the linear function (y = mx + c)
def linear_func(x, m, c):
    return m * x + c


def generate_basis_states(num_qubits):
    basis_states = []
    for state in itertools.product([0, 1], repeat=num_qubits):
        basis_states.append(state)
    return basis_states



def HaarOrbit(state: np.ndarray, N: int, Nl: int):
    """

    For a given input state (vector) produce an output state (vector) that went through a random unitary evolution
    using local gates. The random unitary gates are: Hadamard, S phase gate, and CNOT gate. From this set of gates 
    we choose randomly and apply to the state. We apply N-1 for each neghboring pair of spins.
    We repeat that Nl times.

    HERE APPLY HAAR RANDOM STATE

    """

    sX, sY, sZ, sI = PauliMatrice()

    for z in range(0,Nl):
        for i in range(0,N-1):

            gate = HaarRandom2gate()[0]

            i_index = random.randint(1, N) 
            j_index = random.randint(1, N) 

            state = ApplyLocalGateB(gate,[i_index,j_index],state,N,2)        


    return state


def HaarRandom1gate():
    """ Generating 2x2 random unitary matrix (from 1. Zyczkowski K, Kus M. Random unitary matrices. J Phys A: Math Gen [Internet]. 1994 Jun [cited 2021 Mar 26];27(12):4235–45. ) """ 
    alpha, psi, chi = np.random.uniform(low=0.0, high=2*np.pi, size=3)
    phi = np.arcsin(np.sqrt(np.random.uniform(low=0.0, high=1.0)))

    res = np.zeros((2,2),dtype=np.complex64)
    res[0,0] = np.exp(1.0j*psi)*np.cos(phi)
    res[0,1] = np.exp(1.0j*chi)*np.sin(phi)
    res[1,0] = -np.exp(-1.0j*chi)*np.sin(phi)
    res[1,1] = np.exp(-1.0j*psi)*np.cos(phi)

    return np.exp(1.0j*alpha) * res


def HaarRandom2gate():
    """ the total 4x4 sized random unitary actually is constructed from a tensor product of two 2x2 tensors multiplied with a tensor of 4x4 which is constructed directly from the three nontricial gates sigma_x sigma_x, sigma_y sigma_y, sigma_z sigma_z"""
    U1 = HaarRandom1gate()
    U2 = HaarRandom1gate()

    ridx = np.random.randint(3)

    sX, sY, sZ, sI = PauliMatrice()

    dt = 0.3

    if (ridx == 0):
        Uxx = linalg.expm(1.0j*dt*np.kron(sX,sX))
        return np.kron(U1, U2) @ Uxx, ridx
    elif (ridx == 1):
        Uyy = linalg.expm(1.0j*dt*np.kron(sY,sY))
        return np.kron(U1, U2) @ Uyy, ridx
    else:
        Uzz = linalg.expm(1.0j*dt*np.kron(sZ,sZ))
        return np.kron(U1, U2) @ Uzz, ridx




