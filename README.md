# FabricSense AI: Hyperspectral Textile Categorization
Hyperspectral textile classification using deep learning for automated recycling


## Goal
To simplify automated sorting for reuse facilities, textile classification must reach **95%+ accuracy**.

## Issue
- Recycling is ineffective since garment tags are frequently erroneous or missing.  
- Manual sorting produces a lot of textile waste because it is labor-intensive and prone to mistakes.  

## Remedy
- Create a **deep learning + hyperspectral imaging pipeline** to classify textiles at the pixel level.  
- Validation: initial lab tests show >95% accuracy.  
- Real-time sorting through integration with a conveyor belt system.  

## Workflow Pipeline
![Workflow Pipeline](path/to/your/infographic.png)  
*(Insert infographic about your pipeline here)*  

## Dataset
- Source: **DeepTextile Dataset (BSD License)**  
- Sample file included: `group_5_3_3/Joann_Fab_100Cotton_aggr_5_3_3_0_df.csv`  
- Full dataset available at: [DeepTextile GitHub](https://github.com/danikagupta/DeepTextile)

### Composition of the Dataset
- **15 fabrics** across 5 categories: Cotton, Polyester, Nylon, Cotton/Poly blends, and Poly/Spandex blends.  
- **224 wavelength bands** per pixel with SPECIM FX-17 hyperspectral camera.  
- Aggregated pixel data (such as 3x3 grids) for more effective processing.  

## Significant Events (Milestones)
- [x] Create a Jira project space  
- [x] Create a pipeline for spectral data preprocessing  
- [ ] Train baseline classifiers (SVM, CNN, Random Forest)  
- [ ] Make the deep learning model blend-friendly  
- [ ] Install the prototype on conveyor belt hardware  
- [ ] Reach >95% sorting accuracy in real time  

## Hazards and Reduction
- **Risk:** The model incorrectly classifies blended fabrics → **Mitigation:** increase dataset’s variety of blends.  
- **Risk:** Real-time inference too slow → **Mitigation:** optimize PyTorch inference with quantization.  
- **Risk:** Hardware integration challenges → **Mitigation:** start with a smaller belt system prototype.  

## Supplementary Documents
- Studies on hyperspectral textile classification  
- Dataset documentation (CSV sample + README)  
- Model architecture diagrams (flowcharts, PyTorch code snippets)  
- Experiment logs (confusion matrices, accuracy reports)  
- Deployment plan (integration procedures with sorting belt hardware)  

## Reference
If you use the dataset, please cite:  

@misc{gupta2024deeptextile,  
  title={DeepTextile: NIRS Dataset for Textile Classification},  
  author={Danika Gupta},  
  year={2024},  
  howpublished={\url{https://github.com/danikagupta/DeepTextile}},  
  note={Accessed: \today}  
}

## License
This project uses the **DeepTextile dataset released under BSD License**.  
All code and documentation in this repository are released under [Your License Choice].

## Contact
- **Name:** Anshum Pal  
- **Email:** anshum.pal04@gmail.com  
- **Phone:** +91 9370930133  

## Acknowledgment
We gratefully acknowledge the creators of the **DeepTextile dataset** (Danika Gupta and collaborators) and SPECIM for providing access to the FX-17 hyperspectral imaging system.
