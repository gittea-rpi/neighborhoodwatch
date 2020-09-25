# neighborhoodwatch
An implementation of mix-up where training pairs are convex combinations of k-nearest neighbors, with weights sampled from a Dirichlet distribution. 
As with mix-up, the implementation replaces standard pytorch Datasets with a Dataset that forms these convex combinations. The ConvexHullDataset is 
intended to be a drop-in replacement for map-style Datasets.
