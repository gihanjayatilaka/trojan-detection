echo "START OF SHELL SCRIPT"
cd /vulcanscratch/gihan/trojan/
conda activate keras
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-0\
                    --poisonSampleCount 1000\
                    --dataset mnist\
                    --experimentType shuffled\
                    --modelSaveFile log-exp-20221214a-0-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-1\
                    --poisonSampleCount 1000\
                    --dataset mnist\
                    --experimentType fullBatch\
                    --modelSaveFile log-exp-20221214a-1-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-2\
                    --poisonSampleCount 1000\
                    --dataset mnist\
                    --experimentType percentageOfBatch\
                    --modelSaveFile log-exp-20221214a-2-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-3\
                    --poisonSampleCount 1000\
                    --dataset cifar10\
                    --experimentType shuffled\
                    --modelSaveFile log-exp-20221214a-3-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-4\
                    --poisonSampleCount 1000\
                    --dataset cifar10\
                    --experimentType fullBatch\
                    --modelSaveFile log-exp-20221214a-4-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-5\
                    --poisonSampleCount 1000\
                    --dataset cifar10\
                    --experimentType percentageOfBatch\
                    --modelSaveFile log-exp-20221214a-5-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-6\
                    --poisonSampleCount 2000\
                    --dataset mnist\
                    --experimentType shuffled\
                    --modelSaveFile log-exp-20221214a-6-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-7\
                    --poisonSampleCount 2000\
                    --dataset mnist\
                    --experimentType fullBatch\
                    --modelSaveFile log-exp-20221214a-7-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-8\
                    --poisonSampleCount 2000\
                    --dataset mnist\
                    --experimentType percentageOfBatch\
                    --modelSaveFile log-exp-20221214a-8-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-9\
                    --poisonSampleCount 2000\
                    --dataset cifar10\
                    --experimentType shuffled\
                    --modelSaveFile log-exp-20221214a-9-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-10\
                    --poisonSampleCount 2000\
                    --dataset cifar10\
                    --experimentType fullBatch\
                    --modelSaveFile log-exp-20221214a-10-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-11\
                    --poisonSampleCount 2000\
                    --dataset cifar10\
                    --experimentType percentageOfBatch\
                    --modelSaveFile log-exp-20221214a-11-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-12\
                    --poisonSampleCount 5000\
                    --dataset mnist\
                    --experimentType shuffled\
                    --modelSaveFile log-exp-20221214a-12-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-13\
                    --poisonSampleCount 5000\
                    --dataset mnist\
                    --experimentType fullBatch\
                    --modelSaveFile log-exp-20221214a-13-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-14\
                    --poisonSampleCount 5000\
                    --dataset mnist\
                    --experimentType percentageOfBatch\
                    --modelSaveFile log-exp-20221214a-14-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-15\
                    --poisonSampleCount 5000\
                    --dataset cifar10\
                    --experimentType shuffled\
                    --modelSaveFile log-exp-20221214a-15-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-16\
                    --poisonSampleCount 5000\
                    --dataset cifar10\
                    --experimentType fullBatch\
                    --modelSaveFile log-exp-20221214a-16-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-17\
                    --poisonSampleCount 5000\
                    --dataset cifar10\
                    --experimentType percentageOfBatch\
                    --modelSaveFile log-exp-20221214a-17-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-18\
                    --poisonSampleCount 10000\
                    --dataset mnist\
                    --experimentType shuffled\
                    --modelSaveFile log-exp-20221214a-18-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-19\
                    --poisonSampleCount 10000\
                    --dataset mnist\
                    --experimentType fullBatch\
                    --modelSaveFile log-exp-20221214a-19-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-20\
                    --poisonSampleCount 10000\
                    --dataset mnist\
                    --experimentType percentageOfBatch\
                    --modelSaveFile log-exp-20221214a-20-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-21\
                    --poisonSampleCount 10000\
                    --dataset cifar10\
                    --experimentType shuffled\
                    --modelSaveFile log-exp-20221214a-21-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-22\
                    --poisonSampleCount 10000\
                    --dataset cifar10\
                    --experimentType fullBatch\
                    --modelSaveFile log-exp-20221214a-22-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-23\
                    --poisonSampleCount 10000\
                    --dataset cifar10\
                    --experimentType percentageOfBatch\
                    --modelSaveFile log-exp-20221214a-23-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-24\
                    --poisonSampleCount 20000\
                    --dataset mnist\
                    --experimentType shuffled\
                    --modelSaveFile log-exp-20221214a-24-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-25\
                    --poisonSampleCount 20000\
                    --dataset mnist\
                    --experimentType fullBatch\
                    --modelSaveFile log-exp-20221214a-25-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-26\
                    --poisonSampleCount 20000\
                    --dataset mnist\
                    --experimentType percentageOfBatch\
                    --modelSaveFile log-exp-20221214a-26-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-27\
                    --poisonSampleCount 20000\
                    --dataset cifar10\
                    --experimentType shuffled\
                    --modelSaveFile log-exp-20221214a-27-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-28\
                    --poisonSampleCount 20000\
                    --dataset cifar10\
                    --experimentType fullBatch\
                    --modelSaveFile log-exp-20221214a-28-savedModel.h5;
            
python create-trojan.py\
                    --loggerPrefix log-exp-20221214a-29\
                    --poisonSampleCount 20000\
                    --dataset cifar10\
                    --experimentType percentageOfBatch\
                    --modelSaveFile log-exp-20221214a-29-savedModel.h5;
            
echo "END OF SHELL SCRIPT"
