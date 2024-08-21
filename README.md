# Matilda: Multi-task learning from single-cell multimodal omics

Matilda is a multi-task framework for learning from single-cell multimodal omics data. Matilda leverages the information from the multi-modality of such data and trains a neural network model to simultaneously learn multiple tasks including data simulation, dimension reduction, visualization, classification, and feature selection.

<img width=100% src="https://github.com/liuchunlei0430/Matilda/blob/main/img/main.jpg"/>

For more details, please check out our [publication](https://academic.oup.com/nar/article/51/8/e45/7076464).


# Directory structure

```
.
├── main                      # Main Python package
├── data                      # Data files
├── qc                        # Method evaluation 
├── img                       # Main figure
├── environment_matilda.yaml  # Reproducible Python environment via conda
├── LICENSE
└── README.md
```

# Getting started

Please checkout the documentations and tutorials at https://matil.readthedocs.io/en/latest/.

# Contact

If you found a bug, please use the issue [tracker](https://github.com/PYangLab/Matilda/issues).

# Citation
If you use matilda in your research, please consider citing
```
@article{liu2023multi,
  title={Multi-task learning from multimodal single-cell omics with Matilda},
  author={Liu, Chunlei and Huang, Hao and Yang, Pengyi},
  journal={Nucleic acids research},
  volume={51},
  number={8},
  pages={e45--e45},
  year={2023},
  publisher={Oxford University Press}
}
```

## License

This project is covered under the Apache 2.0 License.

