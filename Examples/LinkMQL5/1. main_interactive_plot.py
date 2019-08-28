from multiprocessing import Process
import time
import sys
import matplotlib.pyplot as plt
import numpy as np

def demo():
    i = 0
    processes = []
    while True:
        i += 1
        s = time.time()
        while time.time() - s < 5:
            print ('HA'),
            sys.stdout.flush()
            
        def do_something():
            figno = i
            f = plt.figure()
            # Normally this will always be "Figure 1" since it's the first
            # figure created by this process. So do something about it.
            f.canvas.set_window_title('My stupid plot number %d' % i)
            arr = np.random.uniform(size=(50, 50))
            plt.imshow(arr)
            plt.show()
        p = Process(None, do_something)
        processes.append(p) # May want to do other things with objects
        p.start()

if __name__ == "__main__":
    demo()