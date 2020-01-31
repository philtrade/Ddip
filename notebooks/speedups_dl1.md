## Speedup observed

Hardware: 3 GTX 1080ti on PCI3 x16,x16,x8 respectively, 32GB main memory, 1TB SSD, CPU: Xeon E5-1620 v4

   Notebook       [3-GPUs timing]  [Single-GPU timing]          LOC
* lesson3-CamVid: [3:30,4:24,12:00,12:52] [7:33,9:12,31:50,33:40]    6
* lesson3-planet: [3:20,3:45,6:15,7:30]   [4:20,5:35,14:35,18:30]    4
* lesson4-collab: Training time too short
* lesson4-tabular: Training time too short, but conversion is simple, only 1 cell, 3 lines.
* lesson5-sgd-mnist: Training time too short
* lesson6-pets-more: [1:06, 0:48,1:30,]      [1:09, 0:48, 1:30]
* lesson6-rossmann: [1:45,1:45,1:45]        [3:00,      4]
* lesson7-superres-imagenet [4:17]  [10:50]
* lesson7-wgan Worse with DDP ! [13:30] [4:41]


