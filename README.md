# CL3: Generalization of Contrastive Loss for Lifelong Learning

Code for [CL3: Generalization of Contrastive Loss for Lifelong Learning](https://arxiv.org/abs/2106.14413). 
Our code is based on the implementation of [Co^2L: Contrastive Continual Learning](https://github.com/chaht01/co2l). 

If you find this code useful, please reference in our paper:

```
@article{roy2023cl3,
  title={CL3: Generalization of Contrastive Loss for Lifelong Learning},
  author={Roy, Kaushik and Simon, Christian and Moghadam, Peyman and Harandi, Mehrtash},
  journal={Journal of Imaging},
  volume={9},
  number={12},
  pages={259},
  year={2023},
  publisher={MDPI}
}
```

# Instruction

Different from other continual learning methods, CL3 needs pre-training part for learning representations since CL3 is a contrastive representation learning based distillation strategy. Thus, you can get the results reported on our paper from linear evaluation with pre-trained representations. Please follow below two commands. 

## Representation Learning
```
python main.py --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --epochs 100 --start_epoch 500 --learning_rate 0.5 --temp 0.5  --cosine --syncBN
```

## Linear Evaluation
```
python main_linear_buffer.py --learning_rate 1 --target_task 4 --ckpt ./save_random_200/cifar10_models/cifar10_32_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.5_momentum_1.000_trial_0_500_100_1.0_cosine_warm/ --logpt ./save_random_200/logs/cifar10_32_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.5_momentum_1.000_trial_0_500_100_1.0_cosine_warm/
```

# Issue

If you have troulbe with NaN loss while training representation learning, you may find solutions from [SupCon issue page](https://github.com/HobbitLong/SupContrast/issues). Please check your training works perfectly on SupCon first. 
