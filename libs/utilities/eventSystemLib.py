#########################################################3
############### BASIC MATH ##############################
##########################################################
## Library with basic mathematical functions 
# These funcitons expect a price sequences:
#     has to be a np.array [Nsamples][Nsec]
import get_data_lib as gdl
import datetime as dt
class CEvent( object ):
    # Generic event to use with EventDispatcher.
    def __init__(self, event_type, data = None, eventID = None):
        #The constructor accepts an event type as string and a data dictionary
        self._type = event_type     # Type of the event for the Dispatcher
        self._eventID = eventID # ID of the handler if we know it   
        self._timeStamp = dt.datetime.now() # TimeStamp when the event was triggered
        self._data = data       # Extra data for the Handlers
        
    def get_type(self):
        return self._type
    def get_data(self):
        return self._data
    def get_timeStamp(self):
        return self._timeStamp
    def get_ID(self):
        return self._eventID

class CHandler( object ):
    # Generic event to use with EventDispatcher.
    def __init__(self, handlerID, handlerFunc):
        #The constructor accepts an event type as string and a data dictionary
        self._handlerFunc = handlerFunc     # Type of the event for the Dispatcher
        self._handlerID = handlerID # ID of the handler if we know it   

    def get_func(self):
        return self._handlerFunc
    def get_ID(self):
        return self._handlerID
        
class CEventDispatcher( object ):
    # Generic event dispatcher which listen and dispatch events
    def __init__(self):
        # We store the pointer to the handlers in 2 dictionaries
        # One by type, the other one by EventID
        # Handlers associated to EventIDs. 
        self.eventID_handlers = dict()
        self.eventType_handlers = dict()
        
    def __del__(self):
        # Remove all listener references at destruction time
        self.eventID_handlers = None
        self.eventType_handlers = None

    def has_listener(self, event_type):
        # Return true if listener is register to event_type
        # Check for event type and for the listener
        if event_type in self._handlers.keys():
            return listener in self._handlers[ event_type ]
        else:
            return False

    def dispatch_event(self, event):
        # Dispatch an instance of Event class
        # Dispatch the event to all the associated listeners
        listeners = []
        ## Case 1: The event has a specific ID
        eventID = event.get_ID() 
        if (type(eventID) != type(None)):
            if (eventID in self.eventID_handlers.keys()):
                listeners = self.eventID_handlers[eventID]
            
            else:
                print "No handler found for this enventID: %s"%eventID
        
        if (len(listeners) == 0):
            event_type = event.get_type()
            if (event_type in self.eventType_handlers.keys()):
                listeners = self.eventType_handlers[event_type]
            else:
                 print "No handler found for this enventType: %s"%event_type
            
        # Fire the listeners !
        for listener in listeners:
            listener.get_func()( event.get_data() )

    def add_event_listener(self, event_type, handlerObj, eventID = None):
        # Add an event listener for an event type
        ## Add the handler to the event type
        listeners = self.eventType_handlers.get( event_type, [] )
        listeners.append( handlerObj )
        self.eventType_handlers[ event_type ] = listeners
        
        # Add listener to the event ID
        if (type(eventID) != type(None)):
            listeners = self.eventID_handlers.get( eventID, [] )
            listeners.append( handlerObj )
            self.eventID_handlers[ eventID ] = listeners
            
    def remove_event_listener(self, event_type, listener, eventID = None):
        # Remove event listener.
        # Remove the listener from the event type
        if self.has_listener( event_type, listener ):
            listeners = self._events[ event_type ]
            if len( listeners ) == 1:
                # Only this listener remains so remove the key
                del self._events[ event_type ]
            else:
                # Update listeners chain
                listeners.remove( listener )
                self._events[ event_type ] = listeners

######### READY TO USE HANDLER FUNCTIONS ############
def download_handler_func(d):
    # This handler downloads the secutity
    if (d["src"] == "Google"):
        TD = gdl.download_TD_google (d["symbolID"], period = d["period"], 
                                     timeInterval = "15d")
    # Create the information for the next handler from the previous info
    d["TD"] = TD
    print TD.iloc[[0,-1]]
    