# SKNet
SKNet是由SK Module模块堆叠而成的neural network。该network的neural unit具备了select kernel的function。该function来源于人类的视觉神经元。这种操作与inception net的多分支操作类似，而且inception net也能够实现select kernel的function。当kernel中对应某一分支的feature map的权重不为0，而且则全为0时，神经元也具备了select kernel的function。但是与inception net的机制不同，SKNet的select kernel function能够根据其Input来自适应决定。所以说，inception net的select kernel function是静态的，而SKNet的select kernel fuunction是动态的。同时，inception net的select kernel function是隐式的，而SKNet的select kernel function是显式的。
# Environment
python 3.7    
pytorch 1.5    
torchvision 0.6    
opencv 3.4  
# Experiment
