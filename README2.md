# AML 2023/2024 Project - Activation Shaping for Domain Adaptation
Repository for "Activation Shaping for Domain Adaptation" project - Advanced Machine Learning Course 2023/2024 @ PoliTo
Alexandre Senouci, s321473, s321473@studenti.polito.it
Paul-Raphaël Spazzola Moracchini, s321955, s321955@studenti.polito.it
Emilie Tran, s321292, s321292@studenti.polito.it


## Running The Experiments
To run the experiments you can use, you can run the launch script `launch_scripts/baseline.sh` :
```
!bash launch_scripts/baseline.sh <target_domain> <mode> <output_folder> <layers> <K> <distrib> <eval>
```
where the argument are :
- target_domain : cartoon, photo, sketch
- mode : baseline, ASHResNet18, ASHResNet18_BA1, ASHResNet18_BA2, ASHResNet18_DA, ASHResNet18_DA_BA1, ASHResNet18_DA_BA2
ASHResNet18 : step2
ASHResNet18_DA : step3
BA1, BA2 : extension 2 (BA1 : no binarization, BA2 : Top-K)
- output_folder : output will be stored in record/<Experiment_mode>/<output_folder>/<target_domain>
- layers : layers on which you want to apply the ASH : [[numLayer,numBlock,numBn],[numLayer,numBlock,numBn],......]
Be careful, do not use space inside the list and sort all the selected layers in the order : [[1,0,1],[1,0,2],[1,1,1],[1,1,2],[2,0,1],[2,0,2],[2,1,1],[2,1,2],[3,0,1],[3,0,2],[3,1,1],[3,1,2],[4,0,1],[4,0,2],[4,1,1],[4,1,2]].
- K : hyper-parameter for Top-K, only used for ASHResNet18_BA2, ASHResNet18_DA_BA2
- distrib : Bernoulli distribution for the random mask, only used for ASHResNet18, ASHResNet18_BA1, ASHResNet18_BA2
- eval : True to activate eval() mode, False to activate eval() mode

It is important to write all the argument even if they do not have effect on the test.

As an example, to reproduce the tests on target domain cartoon for Domain Adaptation : 
```
!bash launch_scripts/baseline.sh cartoon ASHResNet18_DA output [[1,0,1],[4,0,2]] 50 0.5 False
```
(50 and 0.5 are not used but it is important to indicate a value for those arguments)