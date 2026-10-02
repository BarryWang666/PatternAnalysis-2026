# GFNet and CNN for confidence-aware ADNI classification

COMP3710 · s4984179. This research compares a scratch GFNet with a simple
convolutional baseline on prepared 2D MRI slices. The development record below
explains the data, mathematical operations, engineering choices and their
observed consequences. It distinguishes intended experiments from measurements.

## V1 development: patient sampling and aggregated selection

Each network predicts a class for one slice. V1 also constructs scan/patient
summaries and uses weighted scan NLL for selection; the consequences of that
additional objective are investigated as evidence becomes available.

### Recover actual patients and audit the supplied image folders

The source contains 30,520 grayscale JPEG slices from 1,526 acquisitions and
680 patients. Each acquisition contributes twenty prepared slices. The number
before the filename's underscore identifies an acquisition, not a patient;
the `ADNI_..._S_...` token in source metadata supplies the actual patient ID.
Labels map NC to 0 and AD to 1 (the source's AD metadata value is 2).

The supplied train/test folders share 216 patients even though their scan IDs
do not overlap. Different visits from the same person therefore cannot be
assumed independent. The inventory checks diagnosis consistency, dimensions
and recoverable identities before deciding how to divide the data. Folder
names remain provenance and are never model inputs. These counts come from
the saved inventory audit; no clinical diagnosis is inferred from a filename.

## Artificial Intelligence Usage Disclosure

OpenAI Codex assisted with implementation, code organization, interpretation
of saved experiment records and documentation. Model choices and results remain
the author's responsibility. Saved measurements are identified by their run or
job; writing or simplifying code does not reproduce those experiments.

## References

- Rao et al., *Global Filter Networks for Image Classification*, NeurIPS 2021.
  [Paper](https://arxiv.org/abs/2107.00645), [reference code](https://github.com/shakes76/GFNet).
- Wu and He, *Group Normalization*, ECCV 2018.
  [Paper](https://arxiv.org/abs/1803.08494).
- Loshchilov and Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019.
  [Paper](https://arxiv.org/abs/1711.05101).
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
  [Paper](https://proceedings.mlr.press/v70/guo17a.html).
