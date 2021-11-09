import numpy as np
import time
from datetime import timedelta

def _warning(message, category, filename, lineno, file=None, line=None):
    return f'{filename}:{lineno}: {category.__name__}: {message}\n'

def cellpar_to_cell(a, b, c, alpha, beta, gamma):
    """Simplified version of function from ase.geometry.
    
    Unit cell params a,b,c,α,β,γ --> cell as 3x3 ndarray.
    """

    # Handle orthorhombic cells separately to avoid rounding errors
    eps = 2 * np.spacing(90.0, dtype=np.float64)  # around 1.4e-14
    
    cos_alpha = 0. if abs(abs(alpha) - 90.) < eps else np.cos(alpha * np.pi / 180.)
    cos_beta  = 0. if abs(abs(beta)  - 90.) < eps else np.cos(beta * np.pi / 180.)
    cos_gamma = 0. if abs(abs(gamma) - 90.) < eps else np.cos(gamma * np.pi / 180.)
    
    if abs(gamma - 90) < eps:
        sin_gamma = 1.
    elif abs(gamma + 90) < eps:
        sin_gamma = -1.
    else:
        sin_gamma = np.sin(gamma * np.pi / 180.)

    cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sqr = 1. - cos_beta ** 2 - cy ** 2
    
    return np.array([[a,           0,           0], 
                     [b*cos_gamma, b*sin_gamma, 0],
                     [c*cos_beta,  c*cy,        c*np.sqrt(cz_sqr)]])

class ETA:
    """Pass total amount to do on construction,
    then call .update() on every loop."""
    
    _moving_average_factor = 0.3    # epochtime_{n+1} = factor * epochtime + (1-factor) * epochtime_{n}
    
    def __init__(self, to_do, update_rate=100):
        self.to_do = to_do
        self.update_rate = update_rate
        self.counter = 0
        self.start_time = time.time()
        self.tic = self.start_time
        self.time_per_epoch = None
        self.done = False
    
    def _end_epoch(self):
        toc = time.time()
        epoch_time = toc - self.tic
        if self.time_per_epoch is None:
            self.time_per_epoch = epoch_time
        else:
            self.time_per_epoch = ETA._moving_average_factor * epoch_time + \
                                  (1 - ETA._moving_average_factor) * self.time_per_epoch
            
        percent = round(100 * self.counter / self.to_do, 2)
        remaining = int(((self.to_do - self.counter) / self.update_rate) * self.time_per_epoch)
        eta = str(timedelta(seconds=remaining))
        self.tic = toc
        return f'{percent}%, ETA {eta}' + ' ' * 30
    
    def update(self):
        
        self.counter += 1
        
        if self.counter == self.to_do:
            msg = self.finished()
            print(msg, end='\r\n')
            self.done = True
            return
        
        elif self.counter > self.to_do:
            return
        
        if not self.counter % self.update_rate:
            msg = self._end_epoch()
            print(msg, end='\r')
                

    def finished(self):
        total = time.time() - self.start_time
        msg = f'Total time: {round(total,2)}s, ' \
              f'n passes: {self.counter} ' \
              f'({round(self.to_do/total,2)} passes/second)'
        return msg
    
