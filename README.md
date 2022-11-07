<div>
<h2> Autoregressive Slater-Jastrow ansatz for variational Monte Carlo </h2>
<img align="middle" src="_misc/arSJ_sketch.png" width="300" alt="sketch"/>
</div>


So far only the t-V model of spinless fermions is implemented. 

## Key features
 - Get rid of autocorrelation time completely
 - Fast **direct sampling** from a Slater determinant
 - Jastrow factor represented by autoregressive neural network
 - Lowrank update for local kinetic energy preseves cubic scaling 

## Example
See 
```
python3 ./run_autoregVMC.py --help
```
for a full list of options.

## Cite
```
@article{humeniuk2022autoregressive,
  title={Autoregressive neural Slater-Jastrow ansatz for variational Monte Carlo simulation},
  author={Humeniuk, Stephan and Wan, Yuan and Wang, Lei},
  journal={arXiv preprint arXiv:2210.05871},
  year={2022}
}
```
