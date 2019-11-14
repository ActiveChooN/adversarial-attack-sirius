## Adversarial attack Sirius and Yandex workshop project

An attack on image classifiers introduces unpredictable behavior for many computer vision systems. The goal of this project is to investigate a new type of attack, in which completely different images have the same representation in the neurogenic network.

### References

- [Excessive Invariance Causes Adversarial Vulnerability](https://arxiv.org/pdf/1811.00401.pdf)

### Dataset

- MNIST&ImageNet

### Train model

For training model run:

```
python3 main.py --model model_name --dataset dataset_name
```
- Available value for model is example.
- Available value for dataset is MNIST.
- Available value for optimizer is SGD.
- Use --train-batch-size, --test-batch-size, --epochs and --log-interval as training/testing settings.
- Use --lr and --momentum as optimizer settings.
- Use --save-model and --save-path for saving model in the desired path.
- Use the --no-cuda flag to train on the CPU rather than the GPU through CUDA.
- Use --log-level for setting logger level
- Use --metrics for different metrics set for the evaluator
- Set `COMETML_API_KEY` env variable for sending data to [comet.ml](http://comet.ml)  

### Authors

- [Alex Babushkin](https://github.com/ocelaiwo)
- [Diana Zagidullina](https://github.com/dianazagidullina)
- [Dmitry Kalinin](https://github.com/ActiveChooN)
