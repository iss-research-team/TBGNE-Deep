Table 7 Transaction category numbers related to machine tools 
| Transaction category number | Description of transaction content |
| - | - |
| 1090105 |	Metal cutting machine |
| 1090106 |	Metal forming machine |
| 1090107 |	Metal non-cutting and forming processing equipment |
| 109010801 |	Machining center (all CNC machine tools) |
| 109010802 |	CNC metal cutting machine |
| 109010803 |	CNC metal forming machine tools |
| 109010804 | CNC system |
| 1090109 |	Machine tool accessories and auxiliary devices |
| 1090119 |	Valves and faucets |
| 1090120 |	Hydraulic components, systems and devices |
| 1090121 |	Pneumatic components, systems and devices |
| 1090123 |	Bearings and parts |
| 1090124 |	Gears, drive shafts and drive components |
| 1090125 |	Gaskets and similar joint gaskets |

Table 8 DNN classifiers’ parameters for each prediction approaches 
| Prediction approaches | Parameters and their values of DNN classification |
| - | - |
| Local-Deep |	Hidden_layer_sizes=(256,256,256,4)<br>activation= relu<br>Alpha=0.0001 |
| EE+Local-Deep |	Hidden_layer_sizes=(256,256,256,6)<br>activation= relu<br>Alpha=0.001 |
| TBGNE-Deep |	Hidden_layer_sizes=(256,256,256,6)<br>activation= relu<br>Alpha=0.001 |

Table 9 Machine learning classifiers’ parameters for each prediction approaches
| Prediction approaches | Parameters and their values of SVM classification | Parameters and their values of RF classification |
| - | - | - |
| Local-Deep |	C=0.1<br>Loss =squared_hinge<br>Gamma=0.001 |n_estimators=30<br>max_depth=7<br>min_samples_split=3<br>min_samples_leaf=2 |
| EE+Local-Deep |	C=0.1<br>Loss =hinge<br>Gamma=0.001 | n_estimators=20<br>max_depth=5<br>min_samples_split=2<br>min_samples_leaf=1 |
| TBGNE-Deep |	C=0.1<br>Loss =squared_hinge<br>Gamma=0.001 | n_estimators=30<br>max_depth=7<br>min_samples_split=3<br>min_samples_leaf=2 |
