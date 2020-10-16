Training w/ pairs obtained by sampling from amongst the 10 nearest neighbors, using a pretrained ResNet-110 model (to convergence w/ CIFAR using data augmentation), during training, getting precision@1 on validation of around 38% vs a steadily increasing precision@1 on the cvx-mixedup training set (81% when I terminated, after 47 epochs)

Remedies:
 -using real data in addition to the mix-up data to fine-tune, so am not biasing model only towards mix-up data (although why is it able to do so well on the mixup? seems odd)
- potential overfitting, because using the same mixup dataset in each epoch? but this is no different than using real data in each epoch
