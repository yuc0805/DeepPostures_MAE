LP_DATASET_CONFIG = {
    # Baseline, Transfer Learning
    "iwatch": {"in_chans": 3, "nb_classes": 2, "blr": 1e-2,'bs':256,'input_size':[5,100],'weight_decay':5e-2}, 
    
    }


FT_DATASET_CONFIG = {
    # Baseline, Transfer Learning
    "iwatch": {"in_chans": 3, "nb_classes": 2, "blr": 5e-4,'bs':256,'input_size':[3,100],'weight_decay':5e-2}, 
    
    }
