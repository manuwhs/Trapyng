class Event(object):
	def __init__(self, *args):
		self.handlers = set()
		self.args = args
 
	def add(self, fn):
		self.handlers.add(fn)
 
	def remove(self, fn):
		self.handlers.remove(fn)
 
	def __call__(self, *args):
		'''fire the event -- uses __call__ so we can just invoke the object directly...'''
		runtime_args = self.args + args
		for each_handler in self.handlers:
			each_handler(*runtime_args)
 
class ExampleObject(object):
	'''publish start and stop events'''
	def __init__(self):
		self.start = Event('started')
		self.stop = Event('stopped')
 
def example_handler(*args):
	''' reacts to an event'''
	print "example handler fired", args
 
 
fred = ExampleObject()
fred.start.add(example_handler)
fred.stop.add(example_handler)
 
fred.start()
fred.stop('optional extra arg')