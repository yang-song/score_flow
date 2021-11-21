# Maximum Likelihood Training of Score-Based Diffusion Models

This repo contains the official implementation for the paper [Maximum Likelihood Training of Score-Based Diffusion Models](https://arxiv.org/abs/2101.09258)

by [Yang Song](https://yang-song.github.io)\*, [Conor Durkan](https://conormdurkan.github.io/)\*, [Iain Murray](https://homepages.inf.ed.ac.uk/imurray2/), and [Stefano Ermon](https://cs.stanford.edu/~ermon/). Published in NeurIPS 2021 (spotlight).

--------------------

We prove the connection between the Kullback–Leibler divergence and the weighted combination of score matching losses used for training score-based generative models. Our results can be viewed as a generalization of both the de Bruijn identity in information theory and the evidence lower bound in variational inference.

Our theoretical results enable *ScoreFlow*, a continuous normalizing flow model trained with a variational objective, which is much more efficient than neural ODEs. We report the state-of-the-art likelihood on CIFAR-10 and ImageNet 32x32 among all flow models, achieving comparable performance to cutting-edge autoregressive models.

## How to run the code

### Dependencies

Run the following to install a subset of necessary python packages for our code
```sh
pip install -r requirements.txt
```

### Stats files for quantitative evaluation

We provide stats files for computing FID and Inception scores for CIFAR-10 and ImageNet 32x32. You can find `cifar10_stats.npz` and `imagenet32_stats.npz` under the directory `assets/stats` in our [Google drive](https://drive.google.com/drive/folders/1gbDrVrFVSupFMRoK7HZo8aFgPvOtpmqB?usp=sharing). Download them and save to `assets/stats/` in the code repo.

### Usage

Train and evaluate our models through `main.py`. Here are some common options:

```sh
main.py:
  --config: Training configuration.
    (default: 'None')
  --eval_folder: The folder name for storing evaluation results
    (default: 'eval')
  --mode: <train|eval|train_deq>: Running mode: train or eval or training the Flow++ variational dequantization model
  --workdir: Working directory
```

* `config` is the path to the config file. Our config files are provided in `configs/`. They are formatted according to [`ml_collections`](https://github.com/google/ml_collections) and should be quite self-explanatory.

  **Naming conventions of config files**: the name of a config file contains the following attributes:

  * dataset: Either `cifar10` or `imagenet32`
  * model: Either `ddpmpp_continuous` or `ddpmpp_deep_continuous`

*  `workdir` is the path that stores all artifacts of one experiment, like checkpoints, samples, and evaluation results.

* `eval_folder` is the name of a subfolder in `workdir` that stores all artifacts of the evaluation process, like meta checkpoints for supporting pre-emption recovery, image samples, and numpy dumps of quantitative results.

* `mode` is either "train" or "eval" or "train_deq". When set to "train", it starts the training of a new model, or resumes the training of an old model if its meta-checkpoints (for resuming running after pre-emption in a cloud environment) exist in `workdir/checkpoints-meta` . When set to "eval", it can do the following:

  * Compute the log-likelihood on the training or test dataset.
  * Compute the lower bound of the log-likelihood on the training or test dataset.
  * Evaluate the loss function on the test / validation dataset.  
  * Generate a fixed number of samples and compute its Inception score, FID, or KID. Prior to evaluation, stats files must have already been downloaded/computed and stored in `assets/stats`.
	
	When set to "train_deq", it trains a Flow++ variational dequantization model to bridge the gap of likelihoods on continuous and discrete images. Recommended if you want to compete with generative models trained on discrete images, such as VAEs and autoregressive models. `train_deq` mode also supports pre-emption recovery.
	

These functionalities can be configured through config files, or more conveniently, through the command-line support of the `ml_collections` package. 

### Configurations for training
To turn on likelihood weighting, set `--config.training.likelihood_weighting`. To additionally turn on importance sampling for variance reduction, use `--config.training.likelihood_weighting`. To train a separate Flow++ variational dequantizer, you need to first finish training a score-based model, then use `--mode=train_deq`.

### Configurations for evaluation
To generate samples and evaluate sample quality, use the  `--config.eval.enable_sampling` flag; to compute log-likelihoods, use the `--config.eval.enable_bpd` flag, and specify `--config.eval.dataset=train/test` to indicate whether to compute the likelihoods on the training or test dataset. Turn on `--config.eval.bound` to evaluate the variational bound for the log-likelihood. Enable `--config.eval.dequantizer` to use variational dequantization for likelihood computation. `--config.eval.num_repeats` configures the number of repetitions across the dataset (more can reduce the variance of the likelihoods; default to 5).

## Pretrained checkpoints
All checkpoints are provided in this [Google drive](https://drive.google.com/drive/folders/1gbDrVrFVSupFMRoK7HZo8aFgPvOtpmqB?usp=sharing).

Folder structure:

* `assets`: contains `cifar10_stats.npz` and `imagenet32_stats.npz`. Necessary for computing FID and Inception scores.
* `<cifar10|imagenet32>_(deep)_<vp|subvp>_(likelihood)_(iw)_(flip)`. Here the part enclosed in `()` is optional. `deep` in the name specifies whether the score model is a deeper architecture (`ddpmpp_deep_continuous`). `likelihood` specifies whether the model was trained with likelihood weighting. `iw` specifies whether the model was trained with importance sampling for variance reduction. `flip` shows whether the model was trained with horizontal flip for data augmentation. Each folder has the following two subfolders:
	* `checkpoints`: contains the last checkpoint for the score-based model.
	* `flowpp_dequantizer/checkpoints`: contains the last checkpoint for the Flow++ variational dequantization model.

## References

If you find the code useful for your research, please consider citing
```bib
@inproceedings{song2021maximum,
  title={Maximum Likelihood Training of Score-Based Diffusion Models},
  author={Song, Yang and Durkan, Conor and Murray, Iain and Ermon, Stefano},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```

This work is built upon some previous papers which might also interest you:

* Yang Song and Stefano Ermon. "Generative Modeling by Estimating Gradients of the Data Distribution." *Proceedings of the 33rd Annual Conference on Neural Information Processing Systems*, 2019.
* Yang Song and Stefano Ermon. "Improved techniques for training score-based generative models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*, 2020.
* Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. "Score-Based Generative Modeling through Stochastic Differential Equations". *Proceedings of the 9th International Conference on Learning Representations*, 2021.

