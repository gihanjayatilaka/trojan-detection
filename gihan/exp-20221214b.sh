echo "START OF SHELL SCRIPT"
cd /vulcanscratch/gihan/trojan/
conda activate keras
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-0\
                        --poisonSampleCount 5000\
                        --dataset mnist\
                        --diversityFactor None\
                        --modelSaveFile logs/b/log-exp-20221214b-0-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-1\
                        --poisonSampleCount 5000\
                        --dataset mnist\
                        --diversityFactor type:multipleLocations,noLocations:2\
                        --modelSaveFile logs/b/log-exp-20221214b-1-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-2\
                        --poisonSampleCount 5000\
                        --dataset mnist\
                        --diversityFactor type:multipleLocations,noLocations:3\
                        --modelSaveFile logs/b/log-exp-20221214b-2-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-3\
                        --poisonSampleCount 5000\
                        --dataset mnist\
                        --diversityFactor type:locationVariance,locationVariance:2\
                        --modelSaveFile logs/b/log-exp-20221214b-3-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-4\
                        --poisonSampleCount 5000\
                        --dataset mnist\
                        --diversityFactor type:locationVariance,locationVariance:5\
                        --modelSaveFile logs/b/log-exp-20221214b-4-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-5\
                        --poisonSampleCount 5000\
                        --dataset mnist\
                        --diversityFactor type:locationVariance,locationVariance:10\
                        --modelSaveFile logs/b/log-exp-20221214b-5-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-6\
                        --poisonSampleCount 5000\
                        --dataset cifar10\
                        --diversityFactor None\
                        --modelSaveFile logs/b/log-exp-20221214b-6-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-7\
                        --poisonSampleCount 5000\
                        --dataset cifar10\
                        --diversityFactor type:multipleLocations,noLocations:2\
                        --modelSaveFile logs/b/log-exp-20221214b-7-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-8\
                        --poisonSampleCount 5000\
                        --dataset cifar10\
                        --diversityFactor type:multipleLocations,noLocations:3\
                        --modelSaveFile logs/b/log-exp-20221214b-8-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-9\
                        --poisonSampleCount 5000\
                        --dataset cifar10\
                        --diversityFactor type:locationVariance,locationVariance:2\
                        --modelSaveFile logs/b/log-exp-20221214b-9-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-10\
                        --poisonSampleCount 5000\
                        --dataset cifar10\
                        --diversityFactor type:locationVariance,locationVariance:5\
                        --modelSaveFile logs/b/log-exp-20221214b-10-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-11\
                        --poisonSampleCount 5000\
                        --dataset cifar10\
                        --diversityFactor type:locationVariance,locationVariance:10\
                        --modelSaveFile logs/b/log-exp-20221214b-11-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-12\
                        --poisonSampleCount 10000\
                        --dataset mnist\
                        --diversityFactor None\
                        --modelSaveFile logs/b/log-exp-20221214b-12-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-13\
                        --poisonSampleCount 10000\
                        --dataset mnist\
                        --diversityFactor type:multipleLocations,noLocations:2\
                        --modelSaveFile logs/b/log-exp-20221214b-13-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-14\
                        --poisonSampleCount 10000\
                        --dataset mnist\
                        --diversityFactor type:multipleLocations,noLocations:3\
                        --modelSaveFile logs/b/log-exp-20221214b-14-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-15\
                        --poisonSampleCount 10000\
                        --dataset mnist\
                        --diversityFactor type:locationVariance,locationVariance:2\
                        --modelSaveFile logs/b/log-exp-20221214b-15-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-16\
                        --poisonSampleCount 10000\
                        --dataset mnist\
                        --diversityFactor type:locationVariance,locationVariance:5\
                        --modelSaveFile logs/b/log-exp-20221214b-16-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-17\
                        --poisonSampleCount 10000\
                        --dataset mnist\
                        --diversityFactor type:locationVariance,locationVariance:10\
                        --modelSaveFile logs/b/log-exp-20221214b-17-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-18\
                        --poisonSampleCount 10000\
                        --dataset cifar10\
                        --diversityFactor None\
                        --modelSaveFile logs/b/log-exp-20221214b-18-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-19\
                        --poisonSampleCount 10000\
                        --dataset cifar10\
                        --diversityFactor type:multipleLocations,noLocations:2\
                        --modelSaveFile logs/b/log-exp-20221214b-19-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-20\
                        --poisonSampleCount 10000\
                        --dataset cifar10\
                        --diversityFactor type:multipleLocations,noLocations:3\
                        --modelSaveFile logs/b/log-exp-20221214b-20-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-21\
                        --poisonSampleCount 10000\
                        --dataset cifar10\
                        --diversityFactor type:locationVariance,locationVariance:2\
                        --modelSaveFile logs/b/log-exp-20221214b-21-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-22\
                        --poisonSampleCount 10000\
                        --dataset cifar10\
                        --diversityFactor type:locationVariance,locationVariance:5\
                        --modelSaveFile logs/b/log-exp-20221214b-22-savedModel.h5;
                
python create-trojan.py\
                        --loggerPrefix logs/b/log-exp-20221214b-23\
                        --poisonSampleCount 10000\
                        --dataset cifar10\
                        --diversityFactor type:locationVariance,locationVariance:10\
                        --modelSaveFile logs/b/log-exp-20221214b-23-savedModel.h5;
                
echo "END OF SHELL SCRIPT"
