import configparser

class Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))
        
        #Hyper-parameter
        self.epochs = conf.getint("Model_Setup", "epochs")
        self.lr = conf.getfloat("Model_Setup", "lr")
        self.weight_decay = conf.getfloat("Model_Setup", "weight_decay")
        self.k = conf.getint("Model_Setup", "k")
        self.nhid1 = conf.getint("Model_Setup", "nhid1")
        self.nhid2 = conf.getint("Model_Setup", "nhid2")
        self.dropout = conf.getfloat("Model_Setup", "dropout")
        self.beta = conf.getfloat("Model_Setup", "beta")
        self.theta = conf.getfloat("Model_Setup", "theta")
        self.no_cuda = conf.getboolean("Model_Setup", "no_cuda")
        self.no_seed = conf.getboolean("Model_Setup", "no_seed")
        self.seed = conf.getint("Model_Setup", "seed")

        # Dataset
        self.n = conf.getint("Data_Setting", "n")
        self.fdim = conf.getint("Data_Setting", "fdim")
        self.class_num = conf.getint("Data_Setting", "class_num")
        self.structgraph_path = conf.get("Data_Setting", "structgraph_path")
        self.featuregraph_path = conf.get("Data_Setting", "featuregraph_path")
        self.feature_path = conf.get("Data_Setting", "feature_path")
        self.label_path = conf.get("Data_Setting", "label_path")
        self.test_path = conf.get("Data_Setting", "test_path")
        self.train_path = conf.get("Data_Setting", "train_path")





        


