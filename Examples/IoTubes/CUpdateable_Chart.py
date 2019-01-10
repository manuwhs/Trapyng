"""
Class to allow the updatability of a chart by means of a temporizer, without the need for global variables.
The variables are contained in this object !!
"""
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from graph_lib import gl
import numpy as np
import tasks_lib as tl
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
import pandas as pd

import datetime as dt
import utilities_lib as ul
import SQL_lib
import Cemail
import time
from DOCX_lib import CleaningReport
import matplotlib.image as mpimg

from threading import Lock, Thread
import FTP_lib 

"""
Using a class to store data directly
"""
class SQL_DDBB_config():
    DB_NAME = 'iotu41656322575'
    DB_HOST = '160.153.153.41'
    DB_PORT = '3312'
    DB_USER = 'iotu41656322575'
    DB_PASSWORD = 'Ezro|5m8pt9j'
"""
Using a class to store data directly
"""
class email_config():
    user = "iotubes.dk@gmail.com"
    pwd = "Iotubes1"
    recipients = ["manuwhs@gmail.com"]
    """ "maha16bf@student.cbs.dk","Christopher_wild@gmx.net",
                  "jakub.brzosko@best.krakow.pl", "s171869@student.dtu.dk","s172393@student.dtu.dk"]
    """              
    
    #"fogh.engineering@gmail.com"]
    subject = "[IoTubes] Warning on PH value"
    body = "The value of the PH sensor for the claning process is in the warning area"


class Monitor():
    """
    Class that will monitor that the cleananing process is going as it should
    """
    def __init__(self, name = "PH", units = "pH", 
                 desired_value = 0, range_warning = 1, range_stop = 2):
        
        ## Real serial data info 
        self.name = name
        self.units = units
        self.desired_value = desired_value;   # The desired value of the sensor
        self.range_warning = range_warning;   # The range which if crosses we send email
        self.range_stop = range_stop;      # The range which if crosses we stop
        
        
        self.warning_triggered = False  # Flag that tells us if the email has been sent.
        self.stop_triggered = False  # Flag that tells us if the email has been sent.
    
    def check_warning(self, data):
        # This function will send an email if the value of the data is outside the range. Only once 
        if (self.warning_triggered == False):
            if (np.sum(data > self.desired_value + self.range_warning) > 0):
                self.warning_triggered = True;
                return True
            if (np.sum(data < self.desired_value - self.range_warning) > 0):
                self.warning_triggered = True;
                return True
        else:
            return False
    
    def check_stop(self, data):
        # This function will send an email if the value of the data is outside the range. Only once 
        if (self.stop_triggered == False):
            if (np.sum(data > self.desired_value + self.range_stop) > 0):
                self.stop_triggered = True;
                return True
            if (np.sum(data < self.desired_value - self.range_stop) > 0):
                self.stop_triggered = True;
                return True
        else:
            return False
        
class Csource():
    def __init__(self, serial = None):
        
        ## Real serial data info 
        self.serial = serial;
        
        ## Fake data 
        self.fake_indx = 0
        
    def get_data(self):
        """
        This function reads the data from the sensor 
        """
        if (type(self.serial) == type(None)):
            data_value = self.generate_data();
        else: 
    
           # self.serial.flush()
            # Read lines until we reach the end ? 
            
            # If there was no new data
            if (self.serial.in_waiting == 0):
                return None 
            while self.serial.in_waiting:  # Or: while ser.inWaiting():
                data_value = float(self.serial.readline().decode("utf-8").split("\n")[0])
            
        ## Create time stamp 
        time = dt.datetime.now();
        
        return float(data_value), time
    
    def generate_data(self):
        """
        Generates some random data
        """
        Nsamples = 60
        Ndim = 1
        data_sensors = 5.5 + 0.3* np.random.randn(Nsamples,Ndim) + np.array(np.sin(2*np.pi*np.array(range(Nsamples))/(Nsamples/Ndim))).reshape(Nsamples,1);
         
        self.fake_indx += 1;
         
        return data_sensors[self.fake_indx%Nsamples]
     
class CUpdate_chart():
    def __init__(self, name = "IoTubes chart", source = None):
        self.name = name;
        
        #####################       ###########################
        self.source = source; # Sata structure from which we can obtain data
        self.rt_sampling = None;  # Periodic Call temporal to sample data from sources
        self.rt_plotting = None;  # Periodic Call temporal to plot the data, update chart
        
        self.period_sampling = 0.150 #*11;
        self.period_plotting = 0.500 #* 3;
        
        ############### Chart related elements ##################
        self.fig = None;  # Figure where we are plotting
        self.data_axes = None;  # Axes where we have the data
        
        self.show_window = 300; # Number of samples to show in the window
        self.size_buffer = 1000; # Maximum number of samples in the bufer ?
        
        ## Data Related 
        self.data_buffer = [] # Here we store all the aquited values
        self.time_buffer = [] 
        
        self.output_folder = "./IoTubes/"
        self.images_folder = self.output_folder + "images/"
        self.disk_folder = self.output_folder + "sensors/"
        self.reports_folder = self.output_folder + "reports/"
        
        self.output_file_index = 0; # Index to create several files
        
        ul.create_folder_if_needed(self.output_folder);
        
        ##### Specific to the task ######
        
        self.Monitor = Monitor( name = "PH", units = "pH", desired_value = 5.3, range_warning = 2.5, range_stop = 3.2);
        self.Monitor_time = Monitor( name = "Time", units = "minutes", desired_value = 150, range_warning = 10, range_stop = 30);
        self.mySFTP = FTP_lib.SFTP_class()
        self.email_config = email_config;
        
        ######## DDBB data #######
        self.SQL_DDBB_config = SQL_DDBB_config
        
        self.time_now= dt.datetime.now();
        ## Assure uniqueness of id, the starting second when it was creted.
        self.machine_ID = "ID12342"
        self.piping_ID = "Piping_C7"
        self.Responsibles = ["Carsten", "Ole P. Ness"];
        # FDA Cleaning requirements 
        self.FDA_cleaning_procedure = "FDA53531984EX"
        self.cleaning_ID = "CKJ" +  self.machine_ID  +"%i%i%i%i%i%i"%(self.time_now.year,self.time_now.month,self.time_now.day,self.time_now.hour,self.time_now.minute,self.time_now.second)
        
        self.first_time_cleaningID = True;
        
        
        self.plots_data = None
        
        ## Cleaning Report data
        self.report = None
        # Flag for the first time you plot.
        self.first_plot_flag = True 
        
        ## Data lock so that the sampling and the plotting do not access the data at the same time
        self.data_lock = Lock()
        # If the timer function creates different threads for the task, it might be we fuck up charts.
        
        self.plot_lock = Lock() # Not really needed in the end because a Timer() only has a Thread.
        
    def init_figure(self):
        """
        This function initializes the chart, with its widgets and everything
        
        """
        
        button_height = 0.030;
        textbox_length0 = 0.02
        textbox_length1 = 0.04
        textbox_length2 = 0.05
         
        fig = gl.init_figure();
        ## Set the image to full screen
        fig_manager = plt.get_current_fig_manager()
        if hasattr(fig_manager, 'window'):
            fig_manager.window.showMaximized()
    
        data_axes = gl.subplot2grid((1,4), (0,0), rowspan=1, colspan=3)
        
        self.fig = fig; self.data_axes = data_axes;
        
        #### Logo Images !!
        logo_path =  self.output_folder + "images_IoTubes/IoTubes_logo.png"
        image = mpimg.imread(logo_path)
        ax_img = plt.axes([0.725, 0.75, 0.2, 0.2])
        ax_img.imshow(image)
        ax_img.axis("off")
        
        ################## Widgets Axes #####################
        
       
        widgets_x = 0.76
        widgets_x2 = 0.85
        widgets_x3 = 0.90
        
        w1_x, w2_x, w3_x = 0.73, 0.8,0.87
        
        base_y = 0.69
        
        administration_y = base_y
        monitoring_y = administration_y - 0.12
        chart_s_y = monitoring_y - 0.12
        chart_s_y2 = chart_s_y -0.05
        chart_start_stop_y = chart_s_y2 - 0.05
        
        output_y = chart_start_stop_y - 0.12

        
        diff_headline_content =  0.052
        ## Administration ! 
        headlines_x = 0.705
        text = self.fig.text(headlines_x, administration_y + diff_headline_content, 'Administration:', size=20) # ha='center', va='center', size=20)
        
        axbox_machineID = plt.axes([widgets_x, administration_y, textbox_length1, button_height])
        axbox_pipingID = plt.axes([widgets_x2, administration_y, textbox_length1, button_height])

        ### Monitoring
        text = self.fig.text(headlines_x, monitoring_y + diff_headline_content, 'PH Monitoring:', size=20) # ha='center', va='center', size=20)
        axbox_desired_value = plt.axes([widgets_x, monitoring_y, textbox_length0, button_height])
        axbox_range_warning = plt.axes([widgets_x2, monitoring_y, textbox_length0, button_height])
        
        ## Sampling and plotting
        text = self.fig.text(headlines_x, output_y + diff_headline_content, 'Output Generation:', size=20) # ha='center', va='center', size=20)
        axbox_sample_period = plt.axes([widgets_x, chart_s_y, textbox_length1, button_height])
        axbox_plot_period = plt.axes([widgets_x2, chart_s_y, textbox_length1, button_height])
        axbox_Nsamples_show = plt.axes([widgets_x, chart_s_y2, textbox_length1, button_height])
        
        ax_start = plt.axes([widgets_x,chart_start_stop_y, 0.04, button_height])
        ax_stop = plt.axes([widgets_x2, chart_start_stop_y, 0.04, button_height])
        
        ## Output
        text = self.fig.text(headlines_x, chart_s_y + diff_headline_content, 'Sampling and plotting:', size=20) # ha='center', va='center', size=20)
        axsave_disk = plt.axes([w1_x, output_y, 0.055, button_height])
        axsave_DDBB = plt.axes([w2_x, output_y, 0.055, button_height])
        axreport = plt.axes([w3_x, output_y, 0.055, button_height])

        
        ################## Add functionalities ###########################
        
        ################ Chart AXES ################:
        bstop = Button(ax_stop, 'Stop')
        bstop.on_clicked(self.stop_reading_data)
        
        bstart = Button(ax_start, 'Start')
        bstart.on_clicked(self.start_reading_data)
#        bprev.on_clicked(self.auto_update_test)
        
        #### Text input Period  ####
        initial_text = str(int(self.period_sampling * 1000));
        text_box_sample_period = TextBox(axbox_sample_period, 'Sample(ms) ', initial=initial_text)
        text_box_sample_period.on_submit(self.submit_sample_period)
        
        initial_text = str(int(self.period_plotting * 1000));
        text_box_plotting_period = TextBox(axbox_plot_period, 'Plot(ms) ', initial=initial_text)
        text_box_plotting_period.on_submit(self.submit_plotting_period)
        
        #### Text input N samples ####
        initial_text = str(int(self.show_window));
        text_Nsamples_show = TextBox(axbox_Nsamples_show, 'Samples Chart ', initial=initial_text)
        text_Nsamples_show.on_submit(self.submit_show_window)
        
        ################ Data generation widgets ################
        bpsave_disk = Button(axsave_disk, 'Save Disk')
        bpsave_disk.on_clicked(self.save_to_disk)
        
        bpsave_DDBB = Button(axsave_DDBB, 'Save DDBB')
        bpsave_DDBB.on_clicked(self.send_buffer_to_DDBB)
        
        bpsave_report = Button(axreport, 'Report')
        bpsave_report.on_clicked(self.generate_report)
        
        ################ Cleaning input widgets ################
        ## Text input MAchine ID
        initial_text = self.machine_ID
        text_box_machine = TextBox(axbox_machineID, 'Machine ID ', initial=initial_text)
        text_box_machine.on_submit(self.submit_machineID)
    
        initial_text = self.piping_ID
        text_box_piping = TextBox(axbox_pipingID, 'Piping ID ', initial=initial_text)
        text_box_piping.on_submit(self.submit_pipingID)
        
    
        
        ################ MONITORING variables ################
        initial_text = str(self.Monitor.desired_value);
        text_desired_value = TextBox(axbox_desired_value, 'Desired PH ', initial=initial_text)
        text_desired_value.on_submit(self.submit_desired_value)
        
        initial_text = str(self.Monitor.range_warning);
        text_range_warning = TextBox(axbox_range_warning, 'Warning Range ', initial=initial_text)
        text_range_warning.on_submit(self.submit_range_warning)
        
        
        # I think we needed to keep them in memory of they would die
        self.buttons = [bstart, bstop, bpsave_disk,bpsave_DDBB,text_box_machine,
                        text_box_sample_period,text_box_plotting_period,
                        text_Nsamples_show,
                        text_desired_value, text_range_warning,bpsave_report, text_box_piping]


        
        self.initial_text_data = gl.add_text(positionXY = [0.35,0.5], text = r'Waiting for data',fontsize = 30, ax = data_axes)
        
        gl.subplots_adjust(left=.09, bottom=.20, right=.90, top=.90, wspace=.20, hspace=0)
        
        self.monitoring_y = monitoring_y
        
    def send_email(self,data):
#        self.stop_reading_data(None)
#        return None
        #### Add the watking Logo Images !!
        logo_path =  self.output_folder + "images_IoTubes/mail_warning.png"
        image = mpimg.imread(logo_path)
        ax_img = plt.axes([0.88, self.monitoring_y - 0.02, 0.08, 0.08])
        ax_img.imshow(image)
        ax_img.axis("off")
            

        logo_path =  self.output_folder + "images_IoTubes/warning.png"
        image = mpimg.imread(logo_path)
        ax_img = plt.axes([0.77,  0.09, 0.12, 0.12])
        ax_img.imshow(image)
        ax_img.axis("off")
        
        ## Generate image
        folder_images = self.images_folder;
        path_image = folder_images +'Warning.png'
        gl.savefig( path_image,
           dpi = 100, sizeInches = []) # 2*8, 2*3
        
        ############### Send Email ####################
        myMail = Cemail.Cemail(self.email_config.user,self.email_config.pwd,self.email_config.recipients)
        myMail.create_msgRoot(subject = self.email_config.subject + " CID: " + self.cleaning_ID)
        #myMail.set_subject(subject)  # For some reason we can only initilize the Subject
        myMail.add_HTML(self.email_config.body)
            
        myMail.add_image(filedir = path_image, inline = 1)
        
        send_report_flag = True
        if (send_report_flag):
            self.generate_report(None)
            myMail.add_file(self.report_path)
            
        ########## YOU MAY HAVE TO ACTIVATE THE USED OF UNTRUSTFUL APPS IN GMAIL #####
        myMail.send_email()

#        self.start_reading_data(None)
        
    def check_monitoring(self,data):
        """
        Funciton in charged of check the monitoring and act uppon signals
        """
        warning_flag = self.Monitor.check_warning(data);
        if (warning_flag):
            self.send_email(data)
          

    def sample_data(self, extraInfo = None):
            """
            This function aims to saple data needed
            This function is called by the Task Scheduler,
            """
#            print("Gonna sample")
            new_data = self.source.get_data();
            if (type(new_data)==type(None)):
                return True
            data_i, time_i = new_data;
            
    #        if (len(self.data_buffer) == 0):
    #            for i in range(3):
    #                self.data_buffer.append(data_i);
    #                self.time_buffer.append(len(self.data_buffer));
            
            self.data_lock.acquire()
            self.data_buffer.append(data_i);
    #        self.time_buffer.append(len(self.data_buffer));
            
    #        self.data_buffer.append(data_i);
            self.time_buffer.append(time_i);
            self.data_lock.release()
            return True ; ## For the task manager ?
#        print (data_i, time_i )


    def auto_update_test (self,event):
        # I just want to see if the error in plotting comes from Matplotlib or timer
        # So I am just gonna plot real fast this shit with this.
        # IF called in the main program it will not work !! You need to call it from the thread
        for i in range(10):
            self.sample_data()
            self.update_plotting_chart()
            time.sleep(0.1)
        
#        self.stop_reading_data()
    ######### FUNCTION ############
    def update_plotting_chart(self, extraInfo = None):
        """
        This function aims to udate the values in the chart using the new information
        contained in the "information" input structure. 
        This function is called by the Task Scheduler,
        """
        
        self.plot_lock.acquire()
        desired_value = self.Monitor.desired_value   # The desired value of the sensor
        range_warning = self.Monitor.range_warning   # The range which if crosses we send email
        range_stop = self.Monitor.range_stop     # The range which if crosses we stop
        
#        print ("Gonna plot")
        self.data_lock.acquire()
        data, time = np.array(self.data_buffer), np.array(self.time_buffer)
        self.data_lock.release()
        
        if (type(data) == type(None)):
            self.plot_lock.release()
            return True  ; ## For the task manager ?
        if (type(time) == type(None)):
            self.plot_lock.release()
            return True ; ## For the task manager ?
        ## Select the start and end index to plot
        s_indx = max([data.size - self.show_window, 0])
        e_indx = data.size -1
                    
        if(self.first_plot_flag):
            ## Remove the text box
            self.initial_text_data.set_visible(False)
                
            if(len(data) < 2):  # Plot 2 data minimum
                self.plot_lock.release()
                return True ; ## For the task manager ?
            
            self.first_plot_flag = False
            
            ##------------------------------------------
            #### Warning bands 
            ax_aux, plots_data_upper_warning_band = gl.plot([time[s_indx],time[e_indx]], [desired_value + range_warning, desired_value + range_warning],  ax = self.data_axes,
                    color = "y", lw = 3, ls="--", return_drawing_elements = True, legend = ["Warning email"], loc = "upper right"); #, legend = ["Warning area"]
            
            ax_aux, plots_data_lower_warning_band = gl.plot([time[s_indx],time[e_indx]], [desired_value - range_warning, desired_value - range_warning],  ax = self.data_axes,
                    color = "y", lw = 3, ls="--", return_drawing_elements = True);
                                                            
            #### Error bands 
            ax_aux, plots_data_upper_error_band = gl.plot([time[s_indx],time[e_indx]], [desired_value + range_stop, desired_value + range_stop],  ax = self.data_axes,
                    color = "r", lw = 3, ls="--", return_drawing_elements = True, legend = ["Stop"], loc = "upper right"); #, legend = ["Warning area"]
            
            ax_aux, plots_data_lower_error_band = gl.plot([time[s_indx],time[e_indx]], [desired_value - range_stop, desired_value - range_stop],  ax = self.data_axes,
                    color = "r", lw = 3, ls="--", return_drawing_elements = True);
                                                            
                    
            ax_aux, plot_time_series = gl.plot(time[s_indx:e_indx+1], data[s_indx:e_indx+1],  ax = self.data_axes,
                    labels = ["Cleaning Procedure: " + self.cleaning_ID, self.time_now.strftime("%B %d, %Y"), "PH"], color = "k", xaxis_mode = "intraday", return_drawing_elements = True,
                    loc = "upper right");
        
            gl.set_fontSizes(ax = self.data_axes, title = 25, xlabel = 20, ylabel = 20, 
                      legend = 15, xticks = 15, yticks = 15)
            
            ## Save the elements so that we can modify them later
            
            self.plots_data = [plot_time_series[0], plots_data_upper_warning_band[0], plots_data_lower_warning_band[0], plots_data_upper_error_band[0], plots_data_lower_error_band[0]]
            

            
        else:
#            print self.plots_data
            self.plots_data[0].set_xdata(time[s_indx:e_indx+1])
            self.plots_data[0].set_ydata(data[s_indx:e_indx+1])
            
            ## Warning bands
            self.plots_data[1].set_xdata([time[s_indx],time[e_indx]])
            self.plots_data[1].set_ydata([desired_value + range_warning, desired_value + range_warning])
            
            self.plots_data[2].set_xdata([time[s_indx],time[e_indx]])
            self.plots_data[2].set_ydata([desired_value - range_warning, desired_value - range_warning])
            
            ## Error bands
            self.plots_data[3].set_xdata([time[s_indx],time[e_indx]])
            self.plots_data[3].set_ydata([desired_value + range_stop, desired_value + range_stop])
            self.plots_data[4].set_xdata([time[s_indx],time[e_indx]])
            self.plots_data[4].set_ydata([desired_value - range_stop, desired_value - range_stop])
#            gl.set_xlim(ax = self.data_axes, X = time[s_indx:e_indx+1], xmin = np.min(time[s_indx:e_indx+1]), xmax = np.max(time[s_indx:e_indx+1]))
#            gl.set_ylim(ax = self.data_axes, Y = data[s_indx:e_indx+1], ymin =np.min(data[s_indx:e_indx+1]),ymax = np.max(data[s_indx:e_indx+1]))
 

#                gl.set_zoom(X = time[s_indx:e_indx+1],Y = data[s_indx:e_indx+1],xlimPad = [0.2,0.2] ,ylimPad = [0.1, 0.1])
#                gl.set_zoom(X = time[s_indx:e_indx+1],Y = data[s_indx:e_indx+1],xlimPad = [0.2,0.2] ,ylimPad = [0.1, 0.1])
#                gl.set_zoom(X = time[s_indx:e_indx+1],Y = data[s_indx:e_indx+1],xlimPad = [0.2,0.2] ,ylim = [0, 14])
            gl.set_zoom(X = time[s_indx:e_indx+1],Y = data[s_indx:e_indx+1],xlimPad = [0.2,0.2] ,ylim = [0, 10])
            pass
#                self.data_axes.update()
#                self.data_axes.draw(self.plots_data[0])
        plt.draw()
#        l.set_ydata(ydata)
#        ax.set_ylim(np.min(ydata), np.max(ydata))
#        plt.draw()
        
#        self.fig.canvas.draw()
#        #### RETOQUES ########
#        if (len(self.data_buffer) > 1):
#            gl.set_zoom(X = time[s_indx:e_indx+1],Y = data[s_indx:e_indx+1],xlimPad = [0.2,0.2] ,ylimPad = [0.1, 0.1])
        
#
    #    if (update_data.index == 1000):
    #        rt.stop()
    #        information.serial.close()
        self.check_monitoring(data[s_indx:e_indx+1])
        
        self.plot_lock.release()
        return True ; ## For the task manager ?
#    def update_plots():
        
    """
    /////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    ////////////////////////////// WIDGETS FUNCTIONS \\\\\\\\\\\\\\\\\\\\\\\\\\\
    //////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    """
 
    def submit_sample_period(self, text):
        """
        Submit the text of the period text box
        """
        try:
           self.period_sampling = float( text)/1000.0;
        except ValueError:
            print ("Not a float")

    def submit_plotting_period(self, text):
        """
        Submit the text of the period text box
        """
        try:
           self.period_plotting = float( text)/1000.0;
        except ValueError:
            print ("Not a float")
        
        
    def submit_machineID(self, text):
        """
        Submit the text of the period text box
        """
        self.machine_ID = text
    
    def submit_pipingID(self, text):
        """
        Submit the text of the period text box
        """
        self.piping_ID = text
        
    def submit_show_window(self, show_window):
        """
        Submit the text of the period text box
        """
        self.show_window = int(show_window)
        
    def submit_desired_value(self, desired_value):
        """
        Submit the text of the period text box
        """
        self.Monitor.desired_value = float(desired_value)

    def submit_range_warning(self, range_warning):
        """
        Submit the text of the period text box
        """
        self.Monitor.range_warning = float(range_warning)
        
    def submit_range_stop(self, range_stop):
        """
        Submit the text of the period text box
        """
        self.Monitor.range_stop = float(range_stop)
        
    
    def generate_report(self, event):
        """
        This function will generat the report
        """
        print "Generating Report"
        self.report = CleaningReport();
        self.report.load_administration_data([self.machine_ID, self.piping_ID, self.cleaning_ID,self.FDA_cleaning_procedure,self.Responsibles])

        data = [self.time_buffer, self.data_buffer]
        self.report.load_sensors_data(data, column_names = ["PH"])
        self.report.process_cleaning_data([self.Monitor_time ,self.Monitor])
        self.report.generate_images(self.images_folder)
        
        self.report_path = self.reports_folder + "report_"+self.cleaning_ID +".docx"
        self.report.create_document(self.report_path)

        ## Add the report to the web
        self.add_report_to_web()
        
    def stop_reading_data(self,event):
        """
        This function will disable the adquisition and plotting of new data.
        It will distable the periodic function.
        It will be triggered by an event in the chart (click button)
        """
        ## Right now we plot and sample at the same time !!
        
        ## Stop and destroy the interruption object
        if (type(self.rt_plotting) != type(None)):
            self.rt_plotting.stop()
        self.rt_plotting = None
    
        if (type(self.rt_sampling) != type(None)):
            self.rt_sampling.stop()
        self.rt_sampling = None
        
    def start_reading_data(self, event):
        """
        This function will enable the adquisition and plotting of new data.
        This is done by activating the 
        """
        
        if (type(self.rt_plotting) != type(None)):
            self.rt_plotting.stop()
            
        self.rt_plotting = tl.RepeatedTimer(self.period_plotting, self.update_plotting_chart, None)

        if (type(self.rt_sampling) != type(None)):
            self.rt_sampling.stop()
        self.rt_sampling = tl.RepeatedTimer(self.period_sampling, self.sample_data, None)
        
    def save_to_disk(self, event):
        """
        Save the buffer to disk from the UI
        """
        
        self.data_lock.acquire()
        df = pd.DataFrame(
        {'Time': self.time_buffer,
         'Data': self.data_buffer,
    
        });
        self.data_lock.release()
        
        time = dt.datetime.now();
        name = "Machine:" + self.machine_ID + "DataSet (%i,%i,%i,%i,%i,%i)_%i.csv"%(time.year,time.month,time.day,time.hour,time.minute,time.second,self.output_file_index);
        self.output_file_index += 1
        
        df.to_csv(self.disk_folder + name, sep=',')
    
    def send_buffer_to_DDBB(self, event):
        """
        Sends the informaiton in the buffer to the SQL server
        """
        
        cleaning_id = self.cleaning_ID
        self.data_lock.acquire()
        data = [self.time_buffer, self.data_buffer]
        self.data_lock.release()
        ######### STBALICH CONNECTION 
        SQL_lib.update_DDBB_cleaning(cleaning_id, self.SQL_DDBB_config, data , 
                                     column_names = ["PH"], first_time = self.first_time_cleaningID, Monitors = [self.Monitor],
                                     machine_id = self.machine_ID, pipe_id = self.piping_ID, 
                                     procedure = self.FDA_cleaning_procedure,user = self.Responsibles[0])
        self.first_time_cleaningID = False;

    
    ####################################### NOT CALLED BY THE UI ###################
    def start_plotting(self, period = None):
        """
        This function aims to udate the values in the chart using the new information
        contained in the "information" input structure. 
        This function is called by the Task Scheduler,
        """
        if (type(period) != type(None)):
            self.period_plotting = period
            
        if (type(self.rt_plotting) == type(None)):
            self.rt_plotting = tl.RepeatedTimer(self.period_plotting, self.update_data, None)
 
    
 ## Using in the program would be:
    def add_report_to_web(self):
        
        # This function uploads the report on the web via FTP and uploads and HTML file so that we can download it.
    
        # Folder path where to add the report
        server_report_name =  self.report_path.split("/")[-1]##str(int(np.random.randn())*1000000000000000) + ".docx"
    
        local_index_path = "./FTP_added/index.html"
        remote_index_path = "./index.html"
        remote_reports_folder = "./Iotubes_reports/"
        local_report_path = self.report_path
        
        # HTML files modification variables
        self.mySFTP.make_dir(remote_reports_folder)
#        self.mySFTP.make_dir(new_directory)
        # Upload the report file
        self.mySFTP.put_file(local_report_path, remote_reports_folder + server_report_name)

        # Modify the HTML in the website, we download it, modify it and reupload it.
#        get_file(FTP_config_info, source_file = html_file_dir, destination_file = local_file_name)
        
        file = open(local_index_path,'a+') 
    
        included_line = "\n <br><br>" + "<a href='"+remote_reports_folder+server_report_name + "'>" + server_report_name + "</a>" 
        file.write(included_line);
        file.close()
        # Finally modify the file in the DDBB
        self.mySFTP.put_file(local_index_path, remote_index_path)
    
        