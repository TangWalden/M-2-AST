# M²AST: MLP-Mixer-based Adaptive Spatial-Temporal Graph Learning for Human Motion Prediction(Updating)

M²AST, a framework for human motion prediction that leverages the power of MLP-Mixer architectures to adaptively learn spatial-temporal graph representations.

## Abstract
Human motion prediction is a challenging task in human-centric computer vision, involving forecasting future poses based on historical sequences. Despite recent progress in modeling spatial-temporal relationships of motion sequences using complex structured graphs, few approaches have provided an adaptive and lightweight representation for varying graph structures of human motion. Taking inspiration from the advantages of MLP-Mixer, a lightweight architecture designed for learning complex interactions in multi-dimensional data, we explore its potential as a backbone for motion prediction. To this end, we propose a novel MLP-Mixer-based adaptive spatial-temporal pattern learning framework (M$^2$AST). Our framework includes an adaptive spatial mixer to model the spatial relationships between joints, an adaptive temporal mixer to learn temporal smoothness, and a local dynamic mixer to capture fine-grained cross-dependencies between joints of adjacent poses. The final method achieves a compact representation of human motion dynamics by adaptively considering spatial-temporal dependencies from coarse to fine. Unlike the trivial spatial-temporal MLP-Mixer, our proposed approach can more effectively capture both local and global spatial-temporal relationships simultaneously. We extensively evaluated our proposed framework on three commonly used benchmarks (Human3.6M, AMASS, 3DPW MoCap), demonstrating comparable or better performance than existing state-of-the-art methods in both short and long-term predictions, despite having significantly fewer parameters. Overall, our proposed framework provides a novel and efficient solution for human motion prediction with adaptive graph learning.

## Key Features
- **Motion prediction**
- **Adaptive spatial-temporal graph**: 
- **Local dynamic mixer**: 

## Preprint
For a comprehensive understanding of M²AST, its architecture, and its performance benefits, please refer to our preprint available [here](https://assets.researchsquare.com/files/rs-3233962/v1_covered_12dc0e3d-1cfe-406a-8467-0c2efb6245d0.pdf?c=1691547249).

## Citation
If you find our work useful in your research, please consider citing:

```bibtex
@misc{author2024m2ast,
  title={M²AST: MLP-Mixer-based Adaptive Spatial-Temporal Graph Learning for Human Motion Prediction},
  author={Junyi Tang, Yuanwei Liu, Yong Su, Simin An},
  year={2024},
  howpublished={Available at Research Square},
  url={https://assets.researchsquare.com/files/rs-3233962/v1_covered.pdf},
  note={Preprint}
}
