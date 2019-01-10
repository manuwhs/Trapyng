
"""
ELMO !!
"""

from allennlp.models import load_archive
from allennlp.data import DatasetReader
from allennlp.data.dataset import Batch



import os
import sys
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
os.chdir("../../")
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
import import_folders

from squad1_reader import Squad1Reader
load_dataset_from_disk = 0
load_pretrained_BiDAF = 1
build_model_from_scratch = 0

"""
################ LOAD PRETRAINED MODEL ###############
We load the pretrained model and we can see what parts it has and maybe reused if needed.
"""

if (load_pretrained_BiDAF):
    archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz")
    
    # Get the model and the config file
    model = archive.model
    config = archive.config.duplicate()
    
    keys_config = list(config.keys())
    print ("Key config list: ", keys_config)
    for key in keys_config:
        print ("Params of %s"%(key))
        print (config[key])
    ### Get the elements
    ## Data Readers ##
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    ## Vocabulary ##
    vocab = model.vocab 

    """
    ############  Propagate an instance text #############
    """
    instance = dataset_reader.text_to_instance("What kind of test succeeded on its first attempt?", 
                                               "One time I was writing a unit test, and it succeeded on the first attempt.", 
                                               char_spans=[(6, 10)])
    
    print ("Keys instance: ", instance.fields.keys())
    
    # Batch intances and convert to index using the vocabulary.
    instances = [instance]
    dataset = Batch(instances)
    dataset.index_instances(model.vocab)
    
    # Create the index tensor from the vocabulary.
    cuda_device = model._get_prediction_device()
    model_input = dataset.as_tensor_dict(cuda_device=cuda_device)
    
    # Propagate the sample and obtain the loss (since we passed labels)
    outputs = model(**model_input)
    outputs["loss"].requires_grad













