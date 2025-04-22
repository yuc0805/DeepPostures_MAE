DATASET_CONFIG = {
    # Baseline, Transfer Learning
    # "ucihar_7": {"in_chans": 6, "nb_classes": 7, "blr": 1e-2,'bs':12,'img_size':[6,200],'weight_decay':5e-2}, # lin_prob (1e-2, 64)
    # "ucihar_6": {"in_chans": 6, "nb_classes": 6, "blr": 1e-2,'bs':12,'img_size':[6,200],'weight_decay':5e-2}, # ft (1e-3 32)
    # Main Result:
    # Leo #########################
    # "ucihar_7": {"in_chans": 6, "nb_classes": 7, "blr": 1e-2 ,'bs':64,'img_size':[6,200],'weight_decay':5e-2}, # lin_prob (1e-2, 64)
    # "ucihar_6": {"in_chans": 6, "nb_classes": 6, "blr": 1e-2,'bs':64,'img_size':[6,200],'weight_decay':5e-2}, # ft (1e-3 32)
    ##############################
    ##############################
    # Howon fix lr
    "ucihar_7": {"in_chans": 6, "nb_classes": 7, "lr": 1e-3,'bs':50,'img_size':[6,200],'weight_decay':5e-2,'blr':0},
    ###
    #"wisdm": {"in_chans": 3, "nb_classes": 6, "blr": 1e-1,'bs':64,'img_size':[3,200]}, # lin_prob
    # "wisdm": {"in_chans": 3, "nb_classes": 6, "blr": 1e-2,'bs':256,'img_size':[3,200]}, # ORIGINAL #finetune, for baseline, use 1e-1
    #"wisdm": {"in_chans": 3, "nb_classes": 6, "blr": 1e-2,'bs':48,'img_size':[3,200]}, # 94.61, ft
    #
    # "wisdm": {"in_chans": 3, "nb_classes": 6, "blr": 1e-1,'bs':64,'img_size':[3,200],'weight_decay':5e-2}, #TODO: MoCA lin_prob
    "wisdm": {"in_chans": 3, "nb_classes": 6, "blr": 1e-1,'bs':64,'img_size':[3,500],'weight_decay':5e-2}, # Resample to 50Hz
    #
    #"imwsha": {"in_chans":6 , "nb_classes": 11, "blr": 1e-2,'bs':32,'img_size':[6,400]}, # lin_prob 73.02
    #
    #"imwsha": {"in_chans":6 , "nb_classes": 11, "blr": 1e-2,'bs':64,'img_size':[6,400],'weight_decay':5e-2}, # lin_prob 79.37 :)
    #
    "imwsha": {"in_chans":6 , "nb_classes": 11, "blr": 1e-1,'bs':16,'img_size':[6,200],'weight_decay':5e-2}, # resample to 50Hz
    #"imwsha": {"in_chans":6 , "nb_classes": 11, "blr": 1e-3,'bs':250,'img_size':[6,400],'weight_decay':5e-1}, # finetune /1e-2,1e-3 wont work for baseline
    "oppo": {"in_chans":3 , "nb_classes": 4, "blr": 1e-1,'bs':64,'img_size':[3,500],'weight_decay':5e-2}, # resample to 50Hz from 30Hz
    "capture24_4":{"in_chans":3 , "nb_classes": 4, "blr": 1e-2,'bs':256,'img_size':[3,500],'weight_decay':5e-2},
    "capture24_10":{"in_chans":3 , "nb_classes": 10, "blr": 1e-2,'bs':256,'img_size':[3,500],'weight_decay':5e-2},
    }
