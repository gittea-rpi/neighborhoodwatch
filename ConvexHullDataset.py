import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, TensorDataset
import torchvision.datasets as datasets
from torchvision import transforms
from sklearn.metrics.pairwise import distance_metrics
from typing import Any, Callable, Optional, Tuple
import PIL

class ConvexHullDataset(Dataset):
    """Takes in a map-style torch.utils.data.Dataset and generates a new map-style Dataset that 
    consists of samples from the convex hull of points within a certain distance of each other. This
    class can either store the samples, or call the source Dataset each time it is called. """
    
    def __init__(self, 
                 sourcedataset: Dataset, 
                 alpha: Optional[float] = .5, 
                 length: Optional[str] = "auto", 
                 batchsize: Optional[int] = 5000, 
                 store: Optional[bool] = True, 
                 seed: Optional[int] = 0, 
                 averageneighbors: Optional[int] = 2, 
                 maxneighbors: Optional[int] = 100, 
                 distmetric: Optional[str] = "l2", 
                 transform: Optional[Callable] = None,
                 limitdata: Optional[int] = None):
        """
        Args: 
            sourcedataset : an iterable Dataset
            alpha (float): concentration parameter for the Dirichlet distribution
            length (integer or "auto") : length of this Dataset; auto will create same length as input dataset
            batchsize (integer): number of points per batch to use in computing the interpoint distances
            store (binary) : whether to store the data or not
            seed (integer): seed used to generate this Dataset
            averageneighbors (integer): number of neighbors to average over for each sample
            maxneighbors (integer): maximum number of nearest neighbors to store for each point
            distmetric (function): pairwise distance function used to select nearby points, must be a key in sklearn.metrics.pairwise.distance_metrics()
            transform (function): transform for image data sets
        """
        
        self.sourcedataset = sourcedataset
        
        if limitdata is None:
            self.sourcelen = len(self.sourcedataset)
        else:
            self.sourcelen = limitdata
            
        self.alpha = alpha
        if length=="auto":
            self.len = self.sourcelen
        else:
            self.len = length
        
        self.batchsize=batchsize
        self.store = store
        self.seed = seed
        self.averageneighbors = averageneighbors
        self.maxneighbors = maxneighbors
        self.distmetric=distance_metrics()[distmetric]
        self.transform=transform
        
        self.testpoint = sourcedataset[0][0]
        self.testtarget = sourcedataset[0][1]
        
        self.computedistances()
        self.generator = torch.manual_seed(self.seed)
        self.computesamples(self.store)

        # copied from the VisionDataset class to handle transforms
        has_separate_transform = transform is not None
        
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.store:
            obj = self.mydata[idx]
        else:
            obj = self.genitem(idx)
            
        if self.transform is None:
            return obj
        else: 
            return (self.transform(obj[0]), obj[1])
    
    def setaverageneighbors(self, num):
        self.averageneighbors = num
     
    def getpoint(self, data):
        """converts Tensors or PIL Images to numpy arrays"""
        if type(data) == PIL.Image.Image:
            return transforms.PILToTensor()(data).numpy().astype(np.float32)
        else:
            return data.numpy()
    
    def givepoint(self,x):
        """converts numpy array back to a PIL Image or a Tensor"""
        if type(self.testpoint) == PIL.Image.Image:
            return transforms.ToPILImage()(torch.Tensor(x))
        else:
            return torch.Tensor(x)
        
    def genitem(self, idx):
        x = 0*self.getpoint(self.testpoint)
        y = 0*self.testtarget
        for i in np.arange(self.averageneighbors):
            (sx, sy) = self.sourcedataset[self.mixindices[idx, i]]
            sx = self.getpoint(sx)
            x += self.mixcoeffs[idx, i]*sx
            sy += self.mixcoeffs[idx, i]*sy
        return (self.givepoint(x), y)
        
    def computedistances(self):
        """Computes arrays: 
                self.distances of size (sourcelen, maxneighbors) whose ith row contains the distance of the ith datapoint in the source to its nearest neighbors
                self.indices of size (sourcelen, maxneighbors) whose entries are the corresponding indices of the neighbors 
        """
        print("Datatype of feature vectors: %s" % (str(type(self.testpoint))))
        testdatapoint = self.getpoint(self.testpoint)
        datasize = np.prod(testdatapoint.shape)
        datatype = testdatapoint.dtype
        
        self.distances = np.ones((self.sourcelen, self.maxneighbors))*np.Inf
        self.indices = np.zeros((self.sourcelen, self.maxneighbors), dtype=np.int)
        self.maxdistances = np.array([1.0]*self.sourcelen)*np.Inf
        self.maxindices = np.array([0]*self.sourcelen, dtype=np.int)
        
        def replacenearest(row, col, dist):
            maxindex = self.maxindices[row]
            self.indices[row, maxindex] = col
            self.distances[row, maxindex] = dist
            self.maxdistances[row] = np.amax(self.distances[row, :])
            self.maxindices[row] = np.argmax(self.distances[row, :])
            
        def mergedistances(neighbordistances, neighborcolindices, rowindices):
            for rowidx in np.arange(neighbordistances.shape[0]):
                row = rowindices[rowidx]
                for colidx in np.arange(neighbordistances.shape[1]):
                    curdist = neighbordistances[rowidx, colidx]
                    col = neighborcolindices[rowidx, colidx]
                    if curdist < self.maxdistances[row]:
                        replacenearest(row, col, curdist)
                            
        def populatebatch(indices):
            rowpoints = np.empty((len(indices), datasize), dtype=datatype)
            
            currowbatchidx = 0
            for idx in indices:
                rowpoints[currowbatchidx] = self.getpoint(self.sourcedataset[idx][0]).flatten()
                currowbatchidx+=1
            return rowpoints
        
        numrowsplits = np.ceil(self.sourcelen/self.batchsize)
        rowbatch_indices = np.array_split(np.arange(self.sourcelen),numrowsplits)
        
        for rowbatchidx in np.arange(len(rowbatch_indices)):
            print("On row batch %i of %i" % (rowbatchidx+1, len(rowbatch_indices)), flush=True)
            
            rowindices = rowbatch_indices[rowbatchidx]
            rowpoints = populatebatch(rowindices)     
            
            for colbatchidx in np.arange(len(rowbatch_indices)):
                
                colindices = rowbatch_indices[colbatchidx]
                colpoints = populatebatch(colindices)
                
                pairwisedists = self.distmetric(rowpoints, colpoints)
                
                sortindices = np.argsort(pairwisedists, axis=1)
                sortindices = sortindices[:, :np.amin([self.maxneighbors, colpoints.shape[0]])]
                neighborcolindices = np.empty((rowpoints.shape[0], sortindices.shape[1]), dtype=int)
                neighbordistances = np.empty((rowpoints.shape[0], sortindices.shape[1]), dtype=pairwisedists.dtype)
                
                for rowidx in np.arange(rowpoints.shape[0]):
                    neighborcolindices[rowidx] = colindices[sortindices[rowidx]]
                    neighbordistances[rowidx] = pairwisedists[rowidx, sortindices[rowidx]]
                    
                mergedistances(neighbordistances, neighborcolindices, rowindices)
    
    def computesamples(self, store=False):
        self.sampleindices = torch.randint(low=0, high=self.sourcelen, size=(self.len,), generator=self.generator).tolist()
        rng = torch.distributions.dirichlet.Dirichlet(torch.tensor([self.alpha]*self.averageneighbors))
    
        self.mixcoeffs = np.zeros((self.len, self.averageneighbors))
        self.mixindices = np.zeros((self.len, self.averageneighbors), dtype=np.int)
        
        for idx in np.arange(self.len):
            if idx % 1000 == 0:
                print("Computing sample %i of %i" % (idx+1, self.len))
            self.mixcoeffs[idx] = rng.sample().tolist()
            #validindices = np.insert(self.indices[ np.nonzero( np.isfinite( self.distances[idx, :]))], 0, idx)
            validindices = np.insert(self.indices, 0, idx)
            self.mixindices[idx, :] = validindices[torch.randperm(len(validindices), generator=self.generator).numpy()[:self.averageneighbors]]
        
        if store==True:
            self.mydata = [0]*self.len
            for idx in np.arange(self.len):
                self.mydata[idx] = self.genitem(idx)       

if __name__ == "__main__":
    ## Sanity check 
    # To test that the nearest neighbors are being computed correctly, create an artifical dataset
    # compute the nearest neighbors, then check against the nearest neighbors computed by ConvexHullDataset

    from sklearn.neighbors import NearestNeighbors

    n = 1000
    d = 13

    bs = 51
    maxn = 23

    T = torch.Tensor(4*np.random.randn(n, d))
    Y = torch.Tensor(np.random.randn(n))

    nbrs = NearestNeighbors(n_neighbors=maxn).fit(T)
    distances, indices = nbrs.kneighbors(T)

    testDataset = TensorDataset(T, Y)
    testcvxdset = ConvexHullDataset(testDataset, batchsize=bs, maxneighbors=maxn)

    err = 0 # should remain 0
    for idx in np.arange(n):
        err = err + len(set(testcvxdset.indices[idx])-set(indices[idx]))
    print(err)
