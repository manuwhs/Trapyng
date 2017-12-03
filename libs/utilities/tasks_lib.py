#########################################################3
############### BASIC MATH ##############################
##########################################################
## Library with basic mathematical functions 
# These funcitons expect a price sequences:
#     has to be a np.array [Nsamples][Nsec]

from threading import Timer

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.function   = function
        self.interval   = interval
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        # If we are not running already
        if not self.is_running:
            # We set  create the Timer
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

from time import sleep

def hello(name_list):
    for name in name_list:
        print "Hello %s!" % name

#print "starting..."
#rt = RepeatedTimer(1, hello, ["Fruzsi", "Manu"]) # it auto-starts, no need of rt.start()
#try:
#    sleep(10) # your long-running job goes here...
#finally:
#    rt.stop() # better in a try/finally block to make sure the program ends!
