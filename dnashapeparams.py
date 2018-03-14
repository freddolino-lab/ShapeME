import numpy as np

class ShapeParamSeq(object):
    """ Class to hold a sequence of parameters for a single DNA shape parameter

    Attributes:
        name (str): Name of parameter
        params (list): List of parameter values

    """
 
    def __init__(self, name=None, params=None):
        self.name = name
        if params is not None:
            self.params = params
        else:
            self.params = []

    def __iter__(self):
        """ Iterate over each value in the params attribute

        Implemented as a generator that loops over the attribute params

        Returns:
            value (float)
        """
        for param in self.params:
            yield param

    def __getitem__(self, key):
        """ Get item method for taking slices of the attribute params
        
        Args:
            key (int or slice): slice to take out of params

        Returns:
            ShapeParamSeq: a new object with the params attribute sliced
                           according to key
        """
        return ShapeParamSeq(name=self.name, params = self.params[key])

    def add_param(self, param):
        """ Add a new value to the attribute params by appending to the list

        Modifies the object in place

        Modifies:
            params (list)
        """
        self.params = list(self.params)
        self.params.append(param)
        self.params = np.array(self.params)

    def __len__(self):
        """ Length method simply returns the length of the params list

        Returns:
            length (int)
        """
        return len(self.params)

    def get_values(self):
        """ Values method converts and returns params list as a 1d
        numpy array

        Returns:
            np.array: 1d numpy array of length self.length
        """
        return np.array(self.params)

    def normalize_values(self, center, spread):
        """ Method to normalize the values by a center and spread

        Converts values to (values-center)/spread
        """
        values = self.get_values()
        values = (values-center)/spread
        self.params = values

    def unnormalize_values(self, center, spread):
        """ Method to revert the values back to unnormalized by a center and
        spread

        Converts values to (values*spread)+center
        """
        values = self.get_values()
        values = (values*spread) + center
        self.params = values
        

class ShapeParams(object):
    """ Class to hold several shape parameters for the same sequence

    Attributes:
        data (dict): Hold ShapeParamSeq objects by name
        names (list): Names of ShapeParamSeqs stored. determines order for
                      accessing the data in all methods
        vector (np.array): Cache calculation of the vector form. Initially None

    """

    def __init__(self, data=None, names=None):
        if data:
            self.data = data
        else:
            self.data = {}
        if names:
            self.names = names
        else:
            self.names = []

        self.vector= None

    def __len__(self):
        """ Get the length of the object
        
        Length is defined by the length of the FIRST ShapeParamSeq object in the
        data attribute as determined by the names attribute

        Returns:
            length (int)
        """
        return len(self.data[self.names[0]])

    def __iter__(self):
        """ Iters and returns ShapeParamSeq objects by the order in the
        names attribute

        Returns:
            ShapeParamSeq objects
        """
        for name in self.names:
            yield self.data[name]

    def __getitem__(self, key):
        """ Return slices of the ShapeParams object. 
        
        Args:
            key (int or slice): an integer or slice object to slice by
        
        Returns: 
            ShapeParams: a new object with all ShapeParamSeq objects sliced
            according to the key. Names are kept in the same order as the
            attribute names in self
        """
        new_data = {}
        for name in self.names:
            new_data[name] = self.data[name][key] 
        return ShapeParams(data = new_data, names=self.names)
        
    def add_shape_param(self, params):
        """ Add an additional ShapeParamSeq object to the attribute data

        Modifies the object in place. If name already exists, then the
        data is overwritten

        Args:
            params (ShapeParamsSeq): an object with the same length as self

        Modifies:
            names (list): adds the parameter name to the list
            data (dict): adds the param into the dict

        Raises:
            ValueError if the length of the two objects are incompatible 
        """
        if len(self.names) < 1:
            pass
        elif len(params) != len(self):
            while len(params) < len(self):
                params.add_param(np.nan)
#            raise ValueError("Length of params %s does not match length of\
#                              motif %s"%(len(params), len(self)))
        self.data[params.name] = params
        if params.name not in self.names:
            self.names.append(params.name)

    def from_vector(self, names, vector):
        size = len(vector)/len(names)
        for i, name in enumerate(names):
            self.add_shape_param(ShapeParamSeq(name=name, params = vector[i:size]))

    def matrix(self):
        """ Convert the Shape parameter data into a matrix

        Matrix is a np.array that is m x n where m is the number of Shape
        parameters and n is the length of the sequence
        
        Returns:
            m x n numpy array
        """
        matrix = np.zeros((len(self.names), len(self)))
        for i, name in enumerate(self.names):     
            matrix[i,] = self.data[name].params
        return matrix
    
    def as_vector(self, cache=False):
        """ Convert the Shape parameter data into a vector

        vector is a np.array that is 1 x m*n where m is the number of Shape
        parameters and n is the length of the sequence
        
        Returns:
            1 x m*n numpy array
        """
        if cache and self.vector is not None:
            return self.vector
        elif cache:
            mat = self.matrix()
            self.vector = np.reshape(mat, mat.size)
            return self.vector
        else:
            mat = self.matrix()
            return np.reshape(mat, mat.size)

    def windows(self, size, start=0, end= None):
        """ Generator to get equally sized windows of the ShapeParams object
        Args:
            size (int): size of the windows to create
            start (int): start in the sequence to start windows from
            end (int): end in the sequence to stop making windows at
        
        Yields:
            a slice of the ShapeParams object
        """
        if end is None:
            end = len(self)
        last_i = start
        for i in range(start+size, end, size):
            yield self[last_i:i]
            last_i = i

    def sliding_windows(self, size, slide_by=1, start=0, end= None):
        """ Generator to get sliding windows of the ShapeParams object

        Args:
            size (int): size of the windows to create
            slide_by (int): number to slide window by
            start (int): start in the sequence to start windows from
            end (int): end in the sequence to stop making windows at 
        Yields:
            a slice of the ShapeParams object
        """
        if end is None:
            end = len(self)
        start_i = start
        end_i = start+size
        while end_i < (end - slide_by):
            yield self[start_i:end_i]
            start_i = start_i + slide_by
            end_i = start_i + size

    def distance(self, other, vec=False, cache=False):
        """ Calculate the manhattan distance between two ShapeParams

        Args:
            other (ShapeParams object): object to calculate distance from
            vec (boolean): if true then treat other as a np vector
        
        Returns:
            distance (float): manhattan distance
        """
        if vec:
            return np.sum(np.abs(self.as_vector(cache=cache) - other))
        else:
            return np.sum(np.abs(self.as_vector(cache=cache) - other.as_vector()))

if __name__ == "__main__":

    MGW = ShapeParamSeq(name="MGW", params= [4,3,2,1,0,1,2,3,4,5,6,7])
    Roll = ShapeParamSeq(name="Roll", params=[0,1,2,3,4,4,3,2,1,0,2,3])
    HelT = ShapeParamSeq(name="HelT", params= [4,3,2,1,0,1,2,3,4,5,6,7])
    ProT = ShapeParamSeq(name="ProT", params=[0,1,2,3,4,4,3,2,1,0,2,3])
    params = ShapeParams(data = {Roll.name:Roll, MGW.name:MGW, HelT.name: HelT, ProT.name: ProT}, names= ["Roll", "MGW", "HelT", "ProT"])
    for window1 in params.sliding_windows(4, start=1, end=10):
        print window1
    for window1 in params.sliding_windows(4, start=1, end=10):
        print window1
#        for window2 in params.sliding_windows(4, start=1, end=10):
#            print window1.distance(window2)


