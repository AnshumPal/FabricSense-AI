FabricSense AI: Hyperspectral Textile Classification
====================================================

Automated textile classification using hyperspectral imaging and deep learning for recycling and sustainable fashion.

Goal
----
To simplify automated textile sorting in recycling facilities by achieving 95%+ classification accuracy at the pixel level.

Problem
-------
- Garment tags are often missing or incorrect, making manual sorting unreliable.
- Manual sorting is labor-intensive and results in excessive textile waste.

Solution
--------
- Build a deep learning + hyperspectral imaging pipeline to classify textiles automatically.
- Validation: Initial lab tests show >95% accuracy.
- Future application: Real-time sorting integrated with conveyor belt systems.

Workflow Pipeline
-----------------
[Insert Workflow Pipeline Infographic Here]
(Replace with your pipeline infographic showing preprocessing → model → prediction → sorting)

Dataset
-------
Source: DeepTextile Dataset (BSD License)
- Sample included: group_5_3_3/Joann_Fab_100Cotton_aggr_5_3_3_0_df.csv
- Full dataset: https://github.com/danikagupta/DeepTextile

Composition:
- 15 fabrics across 5 categories:
  1. Cotton
  2. Polyester
  3. Nylon
  4. Cotton/Poly blends
  5. Poly/Spandex blends
- 224 hyperspectral bands per pixel using SPECIM FX-17 camera.
- Aggregated pixel data (e.g., 3x3 grids) for more efficient processing.

Milestones
----------
- [x] Create project space (Jira)
- [x] Implement preprocessing pipeline
- [x] Train baseline classifiers (SVM, CNN, Random Forest)
- [ ] Optimize deep learning model for blends
- [ ] Integrate prototype with conveyor belt hardware
- [ ] Achieve >95% real-time sorting accuracy

Risks and Mitigation
--------------------
Risk: Misclassification of blends
Mitigation: Increase dataset variety of blended fabrics

Risk: Slow real-time inference
Mitigation: Optimize PyTorch inference with quantization

Risk: Hardware integration challenges
Mitigation: Start with a smaller prototype belt system

Supplementary Documents
-----------------------
- Hyperspectral textile classification research papers
- Sample dataset CSV + README
- Model architecture diagrams and flowcharts
- Experiment logs (confusion matrices, accuracy reports)
- Deployment plan for conveyor belt integration

Fab ID Mapping
--------------
- 0 → Cotton
- 1 → Cotton/Poly blend
- 2 → Poly/Spandex

Reference
---------
If you use the dataset, please cite:

@misc{gupta2024deeptextile,
  title={DeepTextile: NIRS Dataset for Textile Classification},
  author={Danika Gupta},
  year={2024},
  howpublished={https://github.com/danikagupta/DeepTextile},
  note={Accessed: Today}
}

License
-------
- Dataset: BSD License (DeepTextile dataset)
- Code: BSD 3-Clause License

Contact
-------
- Name: Anshum Pal
- Email: anshum.pal04@gmail.com
- Phone: +91 9370903013

Acknowledgment
--------------
Thanks to Danika Gupta and collaborators for creating the DeepTextile dataset and to SPECIM for providing access to the FX-17 hyperspectral imaging system.