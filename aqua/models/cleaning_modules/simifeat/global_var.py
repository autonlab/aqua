def _init():
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    _global_dict[key] = value


def get_value(key):
    try:
        return _global_dict[key]
    except:
        print('read' + key + 'failed\r\n')



class SimiArgs:
    def __init__(self, num_classes=None,
                 pre_type='CLIP',
                noise_rate=0.6,
                noise_type='manual',
                seed=0,
                G=1,
                k=10,
                cnt=15000,
                max_iter=400,
                local=False,
                loss='fw',
                num_epoch=1,
                min_similarity=0.0,
                Tii_offset=1.0,
                method='rank1') -> None:
        
        self.num_classes = num_classes
        self.pre_type = pre_type
        self.noise_rate = noise_rate
        self.noise_type = noise_type
        self.seed = seed
        self.G = G
        self.k = k
        self.cnt = cnt
        self.max_iter = max_iter
        self.local = local
        self.loss = loss
        self.num_epoch = num_epoch
        self.min_similarity = min_similarity
        self.Tii_offset = Tii_offset
        self.method = method