# CIFAR_KD
CIFAR with Knowledge Distillation 

Knowledge distillation utilizes the performances of larger models to train smaller models with comparable accuracy. 

The teacher model chosen is a `ResNet50` that aims to guide a simply defined CNN to produce an accuracy above base line. 

### Experimental Setup 
```
python src/main.py --config config/config.yaml --train_teacher # Train teacher model first 

python src/main.py --config config/config.yaml # then train student 
```

### Results 