import mysql.connector
from mysql.connector import errorcode
import pandas as pd
import time

def set_summary_entry(cleaning_id, user, procedure, pipe_id, machine_id):
    ## User is the guy responsible for the cleaning
    ## Create an entry in the summary table
    summary_table_name = "cleaning_summary"
    timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
    query=("INSERT INTO `"+ summary_table_name + "`"
        "SET `cleaning_id` = '"+ cleaning_id +"', " +
        "`status` = 'progress', " +
        "`user` = '"+ user +"', " +
        "`datetime` = '"+ timestamp +"', " +
        "`procedure` = '"+ procedure +"', " +
        "`machine_id` = '"+ machine_id +"', " +
        "`pipe_id` = '"+ pipe_id + "'")
    return query

def update_table_status(cleaning_id,status = "pass"):
    ## Updates the pass or fail status of the cleaning process
    summary_table_name = "cleaning_summary"
    query=("UPDATE `"+ summary_table_name +
        "`SET `status` = '"+ status +
        "' WHERE `"+ summary_table_name +"`.`cleaning_id` = '"+ cleaning_id + "'")
    return query

def create_cleaning_table(cleaning_id = "CCCCCC"):
    query = ("CREATE TABLE `"+cleaning_id+"` ("
    "  `TS` TIMESTAMP NOT NULL,"     # Time stamp of cleaning
    "  `Temp` FLOAT(6,3)  NOT NULL,"    # Temperature (usually C) NULLABLE
    "  `PH` FLOAT(6,3)  NOT NULL,"    # PH (usually C) NULLABLE
    "  `Pressure` FLOAT(6,3)  NOT NULL,"    # PH (usually C) NULLABLE
    "  `Conductivity` FLOAT(6,3)  NOT NULL,"    # PH (usually C) NULLABLE
    "  PRIMARY KEY (`TS`)"
    ") ENGINE=InnoDB")   #  FLOAT(3,3)    TIMESTAMP() 
    
    return query

def delele_table(table_id = ""):
    query = ("DROP TABLE `"+table_id+"`"                                     
    )   
    return query

def create_DDBB(databasename = ""):
    query = ("CREATE DATABASE "+databasename+";"                                     
    )   
    return query

def excute_query(query,cursor, extra_text = "" ):
    try:
        print(extra_text)
        cursor.execute(query)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            print("already exists.")
        else:
            print(err.msg)
    else:
        print("OK")

def add_cleanning_data(cleaning_id, data):
    """
    This funcrion parses the data from the sensors into a SQL query.
    """
    
    column_names = data.columns.values.tolist();  # Temp, PH,Pressure,Conductivity)
    column_names.remove("TS"); # Remove the time index  
    
    print (column_names)
    
    query = "INSERT INTO `"+cleaning_id+ "` ( TS"
    for column_name in  column_names:
        query += ","+column_name
    query += ")VALUES "
    
    for i in range(len(data)):
        # TODO: Change to look over the possible columns 
        query += "('" + data["TS"][i].strftime('%Y-%m-%d %H:%M:%S') +"'"
#        query += "( TO_TIMESTAMP('" + data["TS"][i].strftime('%Y-%m-%d-%H.%M.%S.%f') + "', 'YYYY-MM-DD-hh.mi.ss.ff')"
        for column_name in  column_names:
            query += ",'"+ "%.2f"%data[column_name][i] +"'"
#            query += "','"+ "%.2f"%data["Temp"][i] + "','"+ "%.2f"%data["PH"][i] + "','"+ \
#                 "%.2f"%data["Pressure"][i]+ "','"+ "%.2f"%data["Conductivity"][i]+"')"
        query += ")"
        if(i < len(data)-1):
            query+=","
        else:
             query+=";"
    
    return query

def get_cleanning_data(cleaning_id):
     query = "SELECT * FROM  `"+cleaning_id+ "`"
     return query

## &////////////////////// FUNCTIONS WITH EVERYTHING \\\\\\\\\\\\\\\\\\\\\\\\

def update_DDBB_cleaning(cleaning_id, SQL_DDBB_config, data, column_names = ["Temp","PH","Pressure","Conductivity"], first_time = True, Monitors = [],
                          user = "test_user", procedure = "Dafult_procedure", pipe_id = "Default_pip", machine_id = "Dafult_machine"):
    
    ######### STBALICH CONNECTION 
    cnx = mysql.connector.connect(user=SQL_DDBB_config.DB_USER, password=SQL_DDBB_config.DB_PASSWORD,
                                  host=SQL_DDBB_config.DB_HOST, port = SQL_DDBB_config.DB_PORT,
                                  database=SQL_DDBB_config.DB_NAME)
    cursor = cnx.cursor()

    ## First see if it was created 
    
    if (first_time):
        query = set_summary_entry(cleaning_id, user, procedure, pipe_id, machine_id)
        print query
        excute_query(query,cursor, extra_text = "create summary entry" )
        ## Create table table
        query = create_cleaning_table(cleaning_id = cleaning_id)
        excute_query(query,cursor, extra_text = "create table" )
             
    else:
        ## Drop table
        query = delele_table(table_id = cleaning_id)
        excute_query(query,cursor, extra_text = "droppin table" )
        
        ## Create table table
        query = create_cleaning_table(cleaning_id = cleaning_id)
        excute_query(query,cursor, extra_text = "create table" )
        
    
    data_timestamps, data_sensors  = data
    # column_names = ["Temp","PH","Pressure","Conductivity"] #"TS",
    
    df = pd.DataFrame(data_sensors, columns = column_names)
    df["TS"] = data_timestamps
    
    ## Add the data:
    query = add_cleanning_data(cleaning_id, df)
    print (query)
    excute_query(query,cursor, extra_text = "Uploading data" )
    
    ##### Say if the cleaning was successful or not ####
    status = "pass"
    if(Monitors[0].warning_triggered):
        status = "fail"
        
    query = update_table_status(cleaning_id,status)
    excute_query(query,cursor, extra_text = "Updating status" )
    
    cnx.commit()
    
TABLES = {}
TABLES['employees'] = (
    "CREATE TABLE `employees` ("
    "  `emp_no` int(11) NOT NULL AUTO_INCREMENT,"
    "  `birth_date` date NOT NULL,"
    "  `first_name` varchar(14) NOT NULL,"
    "  `last_name` varchar(16) NOT NULL,"
    "  `gender` enum('M','F') NOT NULL,"
    "  `hire_date` date NOT NULL,"
    "  PRIMARY KEY (`emp_no`)"
    ") ENGINE=InnoDB")

TABLES['departments'] = (
    "CREATE TABLE `departments` ("
    "  `dept_no` char(4) NOT NULL,"
    "  `dept_name` varchar(40) NOT NULL,"
    "  PRIMARY KEY (`dept_no`), UNIQUE KEY `dept_name` (`dept_name`)"
    ") ENGINE=InnoDB")

TABLES['salaries'] = (
    "CREATE TABLE `salaries` ("
    "  `emp_no` int(11) NOT NULL,"
    "  `salary` int(11) NOT NULL,"
    "  `from_date` date NOT NULL,"
    "  `to_date` date NOT NULL,"
    "  PRIMARY KEY (`emp_no`,`from_date`), KEY `emp_no` (`emp_no`),"
    "  CONSTRAINT `salaries_ibfk_1` FOREIGN KEY (`emp_no`) "
    "     REFERENCES `employees` (`emp_no`) ON DELETE CASCADE"
    ") ENGINE=InnoDB")

TABLES['dept_emp'] = (
    "CREATE TABLE `dept_emp` ("
    "  `emp_no` int(11) NOT NULL,"
    "  `dept_no` char(4) NOT NULL,"
    "  `from_date` date NOT NULL,"
    "  `to_date` date NOT NULL,"
    "  PRIMARY KEY (`emp_no`,`dept_no`), KEY `emp_no` (`emp_no`),"
    "  KEY `dept_no` (`dept_no`),"
    "  CONSTRAINT `dept_emp_ibfk_1` FOREIGN KEY (`emp_no`) "
    "     REFERENCES `employees` (`emp_no`) ON DELETE CASCADE,"
    "  CONSTRAINT `dept_emp_ibfk_2` FOREIGN KEY (`dept_no`) "
    "     REFERENCES `departments` (`dept_no`) ON DELETE CASCADE"
    ") ENGINE=InnoDB")

TABLES['dept_manager'] = (
    "  CREATE TABLE `dept_manager` ("
    "  `dept_no` char(4) NOT NULL,"
    "  `emp_no` int(11) NOT NULL,"
    "  `from_date` date NOT NULL,"
    "  `to_date` date NOT NULL,"
    "  PRIMARY KEY (`emp_no`,`dept_no`),"
    "  KEY `emp_no` (`emp_no`),"
    "  KEY `dept_no` (`dept_no`),"
    "  CONSTRAINT `dept_manager_ibfk_1` FOREIGN KEY (`emp_no`) "
    "     REFERENCES `employees` (`emp_no`) ON DELETE CASCADE,"
    "  CONSTRAINT `dept_manager_ibfk_2` FOREIGN KEY (`dept_no`) "
    "     REFERENCES `departments` (`dept_no`) ON DELETE CASCADE"
    ") ENGINE=InnoDB")

TABLES['titles'] = (
    "CREATE TABLE `titles` ("
    "  `emp_no` int(11) NOT NULL,"
    "  `title` varchar(50) NOT NULL,"
    "  `from_date` date NOT NULL,"
    "  `to_date` date DEFAULT NULL,"
    "  PRIMARY KEY (`emp_no`,`title`,`from_date`), KEY `emp_no` (`emp_no`),"
    "  CONSTRAINT `titles_ibfk_1` FOREIGN KEY (`emp_no`)"
    "     REFERENCES `employees` (`emp_no`) ON DELETE CASCADE"
    ") ENGINE=InnoDB")

