from numpy.polynomial import Polynomial
import numpy as np

def round_coordinates(coordinates):
    """Gives the integral rest."""
    coordinates = coordinates - np.floor(coordinates)
    return coordinates

def coordinate_wise_random_rounding(coordinates):
    """Rounds coordinates randonmly."""
    r = round_coordinates(coordinates)
    f = np.array([np.random.choice([c, c-1], 1, p=[1-c, c]) for c in r]).reshape(-1)
    
    rounded_coordinates = coordinates - f
    rounded_coordinates = [int(coeff) for coeff in rounded_coordinates]
    return rounded_coordinates

class CKKS:
    """Basic CKKS encoder to encode complex vectors into polynomials."""
    
    def __init__(self, M:int, scale:float):
        """Initializes with scale."""
        self.xi = np.exp(2 * np.pi * 1j / M)
        self.M = M
        self.create_sigma_R_basis()
        self.scale = scale
        
    @staticmethod
    def vandermonde(xi: np.complex128, M: int) -> np.array:
        """Computes the Vandermonde matrix from a m-th root of unity."""
        
        N = M //2
        matrix = []
        # We will generate each row of the matrix
        for i in range(N):
            # For each row we select a different root
            root = xi ** (2 * i + 1)
            row = []

            # Then we store its powers
            for j in range(N):
                row.append(root ** j)
            matrix.append(row)
        return matrix
    
    def sigma_inverse(self, b: np.array) -> Polynomial:
        """Encodes the vector b in a polynomial using an M-th root of unity."""

        # First we create the Vandermonde matrix
        A = CKKS.vandermonde(self.xi, self.M)

        # Then we solve the system
        coeffs = np.linalg.solve(A, b)

        # Finally we output the polynomial
        p = Polynomial(coeffs)
        return p

    def sigma(self, p: Polynomial) -> np.array:
        """Decodes a polynomial by applying it to the M-th roots of unity."""

        outputs = []
        N = self.M //2

        # We simply apply the polynomial on the roots
        for i in range(N):
            root = self.xi ** (2 * i + 1)
            output = p(root)
            outputs.append(output)
        return np.array(outputs)
    

    def pi(self, z: np.array) -> np.array:
        """Projects a vector of H into C^{N/2}."""

        N = self.M // 4
        return z[:N]


    def pi_inverse(self, z: np.array) -> np.array:
        """Expands a vector of C^{N/2} by expanding it with its
        complex conjugate."""

        z_conjugate = z[::-1]
        z_conjugate = [np.conjugate(x) for x in z_conjugate]
        return np.concatenate([z, z_conjugate])
    
    def create_sigma_R_basis(self):
        """Creates the basis (sigma(1), sigma(X), ..., sigma(X** N-1))."""

        self.sigma_R_basis = np.array(self.vandermonde(self.xi, self.M)).T
    

    def compute_basis_coordinates(self, z):
        """Computes the coordinates of a vector with respect to the orthogonal lattice basis."""
        output = np.array([np.real(np.vdot(z, b) / np.vdot(b,b)) for b in self.sigma_R_basis])
        return output

    def sigma_R_discretization(self, z):
        """Projects a vector on the lattice using coordinate wise random rounding."""
        coordinates = self.compute_basis_coordinates(z)

        rounded_coordinates = coordinate_wise_random_rounding(coordinates)
        y = np.matmul(self.sigma_R_basis.T, rounded_coordinates)
        return y


    def encode(self, z: np.array) -> Polynomial:
        """Encodes a vector by expanding it first to H,
        scale it, project it on the lattice of sigma(R), and performs
        sigma inverse.
        """
        pi_z = self.pi_inverse(z)
        scaled_pi_z = self.scale * pi_z
        rounded_scale_pi_zi = self.sigma_R_discretization(scaled_pi_z)
        p = self.sigma_inverse(rounded_scale_pi_zi)

        # We round it afterwards due to numerical imprecision
        coef = np.round(np.real(p.coef)).astype(int)
        p = Polynomial(coef)
        
        s_Bclean = 10
        N = self.M //2
        self.poly_modulus = Polynomial([1] + [0] * (N-1) + [1])
        self.s = Polynomial(np.random.uniform(low=1, high=s_Bclean, size=(N,)).astype(int)) % self.poly_modulus
        
        return p


    def decode(self, p: Polynomial) -> np.array:
        """Decodes a polynomial by removing the scale, 
        evaluating on the roots, and project it on C^(N/2)"""
        rescaled_p = p / self.scale
        z = self.sigma(rescaled_p)
        pi_z = self.pi(z)
        return pi_z
    
    def encrypt(self, p1: Polynomial) -> dict:
        '''Encrypt a polynomial'''
        
        #R=z[x]/x^n+1
        N = self.M //2
        a_Bclean, e_Bclean = 100, 10
        a_coeff = np.random.uniform(low=1, high=a_Bclean, size=(N,)).astype(int) #uniform
        e_coeff = np.random.randint(low=0, high=e_Bclean, size=(N,)).astype(int) #gaussian
        a1 = Polynomial(a_coeff) % self.poly_modulus#random uniform
        e1 = Polynomial(e_coeff) % self.poly_modulus#random gaussian
        b1 = (p1 - a1*self.s + e1) % self.poly_modulus
        return {'b':b1, 'a':a1}
    
    def HE_add(self, c1:dict, c2:dict) -> dict:
        '''Homomorphic'''
        return {'b':c1['b']+c2['b']% self.poly_modulus, 'a':c1['a']+c2['a']% self.poly_modulus}
    
    def add_decrypt(self, c_add:dict) -> Polynomial:
        return (c_add['b']+c_add['a']*self.s) % self.poly_modulus
    
    def HE_mult(self, p2, c1) -> tuple:
        return (p2*c1['b'] % self.poly_modulus, p2*c1['a'] % self.poly_modulus)
    
    def mult_decrept(self, c_mult:tuple) -> Polynomial:
        B1, B2 = c_mult[0], c_mult[1]
        dec_c_mult = B1 + (B2*self.s) % self.poly_modulus
        dec_c_mult %= self.poly_modulus
        return dec_c_mult / self.scale