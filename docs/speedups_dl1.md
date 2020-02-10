## Speedup observed

Hardware: 3 GTX 1080ti on PCI3 x16,x16,x8 respectively, 32GB main memory, 1TB SSD, CPU: Xeon E5-1620 v4


|    Notebook     | [3-GPUs timing]        | [Single-GPU timing]        | 
|-----------------|:-----------------------|:---------------------------|
| [lesson3-CamVid](lesson3-CamVid.ipynb): |[3:30,4:24,12:00,12:52] | [7:33,9:12,31:50,33:40]    |
| [lesson3-planet](lesson3-planet.ipynb): | [3:20,3:45,6:15,7:30]  | [4:20,5:35,14:35,18:30]    |
| [lesson3-imdb](lesson3-imdb.ipynb): | Accuracy problem in 2nd half | |
| [lesson4-collab](lesson4-collab.ipynb): | Training time too short |  |
| [lesson4-tabular](lesson4-tabular.ipynb):| Training time too short |  |
| [lesson5-sgd-mnist](lesson5-sgd-mnist.ipynb): | Training time too short |   |
| [lesson6-pets-more](lesson6-pets-more.ipynb): | [1:06, 0:48, 1:30]  |   [1:09, 0:48, 1:30] |
| [lesson6-rossmann](lesson6-rossmann.ipynb): | [1:45,1:45,1:45]    |    [3:05, 3:05, 3:05] |
| [lesson7-human-numbers](lesson7-human-numbers.ipynb): | Training time too short | |
| [lesson7-superres-imagenet](lesson7-superres-imagenet.ipynb): | [4:17] | [10:50] | |
| lesson7-wgan: | [13:30/epoch] ***Ouch***!  | [4:41/epoch]


