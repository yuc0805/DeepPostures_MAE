LP_DATASET_CONFIG = {
    # Baseline, Transfer Learning
    "iwatch": {"in_chans": 3, "nb_classes": 2, "blr": 1e-2,'bs':256,'input_size':[3,100],'weight_decay':1e-3}, 
    
    }


FT_DATASET_CONFIG = {
    # Baseline, Transfer Learning
    "iwatch": {"in_chans": 3, "nb_classes": 2, "blr": 1e-3,'bs':256,'input_size':[3,100],'weight_decay':5e-2}, 
    
    }

LP_LONG_DATASET_CONFIG = {
    "iwatch": {"in_chans": 3, "nb_classes": 2, "blr": 1e-2,'bs':4,'input_size':[3,4200],'weight_decay':5e-2}, 
}