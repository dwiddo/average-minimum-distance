import time
from datetime import timedelta

class _ETA:
    """Pass total amount to do on construction,
    then call .update() on every loop."""
    
    _moving_average_factor = 0.3    # epochtime_{n+1} = factor * epochtime + (1-factor) * epochtime_{n}
    
    def __init__(self, to_do, update_rate=1000):
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
            self.time_per_epoch = _ETA._moving_average_factor * epoch_time + \
                                  (1 - _ETA._moving_average_factor) * self.time_per_epoch
            
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
