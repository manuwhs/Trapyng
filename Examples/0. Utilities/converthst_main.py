import import_folders
import utilities_lib as ul
import convert_lib as convl
#updates_folder = "../Hanseatic/MQL4/Files/"
#storage_folder = "./storage/Hanseatic/"

#updates_folder = "../FxPro/MQL4/Files/"
#storage_folder = "./storage/FxPro/"

updates_folder = "../GCI/MQL4/Files/"
storage_folder = "./storage/GCI/"

hst_folder ="../GCI/history/GCI-Demo/" 

#hst_folder ="../FxPro/history/FxPro.com-Demo04/"
#hst_folder ="../Hanseatic/history/HBS-CFD-Server/"

#convl.process_hst(hst_file)
#hst_file ="../GCI/history/GCI-Demo/EURCAD.s1.hst" 

all_paths = ul.get_allPaths(hst_folder)

for path in all_paths:
    convl.process_hst(path)

