class Params(object):
    def __init__(self, **kwargs):
        self.inhibition = None
        self.n_days = 30
        self.record_on_days = [0, 29]
        self.neuron = {
            'N_E': kwargs.get('N_E', 20000),
            'N_I': kwargs.get('N_I', 5000),
            'c': kwargs.get('c', 0.04),
            'g': kwargs.get('g', 12),
            'tau': kwargs.get('tau', 1e-2),
        }
        self.plasticity = {
            'A': kwargs.get('A', 5),
            'x_f': kwargs.get('x_f', 1.5),
            'E_x_f': kwargs.get('E_x_f', -0.133),
        }
        self.perturbation = {
            'lambda': kwargs.get('lambda_', 0),
            'sigma_z': kwargs.get('sigma_z', 0),
        }
        self.omega = {
            'g': kwargs.get('omega_g', 1),
            'o': kwargs.get('omega_o', 0),
        }
        self.debug = kwargs.get('debug', False)

    @classmethod
    def from_dict(cls, d):
        p = cls()
        p.inhibition = d.get('inhibition')
        p.n_days = d.get('n_days')
        p.record_on_days = d.get('record_on_days')
        p.neuron = d.get('neuron')
        p.plasticity = d.get('plasticity')
        p.perturbation = d.get('perturbation')
        p.omega = d.get('omega')
        return p

    def to_dict(self):
        d = dict()
        d['inhibition'] = self.inhibition
        d['n_days'] = self.n_days
        d['record_on_days'] = self.record_on_days
        d['neuron'] = self.neuron
        d['plasticity'] = self.plasticity
        d['perturbation'] = self.perturbation
        d['omega'] = self.omega
        return d

    def update(self, key, val):
        key1, key2 = key.split('.')
        d = getattr(self, key1)
        d[key2] = val
        setattr(self, key1, d)
