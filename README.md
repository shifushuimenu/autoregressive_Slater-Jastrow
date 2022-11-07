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
For a full list of options see
```
python3 ./run_autoregVMC.py --help
```
Run VMC for t-V model on a square 4x4 lattice with 8 spinless fermions and interaction strength V/t=6.0.
training iteractions = 1000, batch size per iteration = 200, num. of samples in measurement phase = 300.
```
python3 ./run_autoregVMC.py 4 4 8 6.0 1000 200 300 --optimizer Adam --seed 42 --optimize_orbitals True
```
## Cite
```
@article{humeniuk2022autoregressive,
  title={Autoregressive neural Slater-Jastrow ansatz for variational Monte Carlo simulation},
  author={Humeniuk, Stephan and Wan, Yuan and Wang, Lei},
  journal={arXiv preprint arXiv:2210.05871},
  year={2022}
}
```
