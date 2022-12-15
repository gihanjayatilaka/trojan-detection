#!/usr/bin/env python
# coding: utf-8

# In[ ]:


EXP = "b"


# In[3]:


hyperParams = {}


hyperParams["poisonSampleCount"]=[1000,2000,5000,10000,20000]
hyperParams["dataset"]=["mnist","cifar10"]
hyperParams["fixedPoisonLocation"]=[1,4,10,15]
hyperParams["experimentType"]=["shuffled","fullBatch","percentageOfBatch"]
hyperParams["diversityFactor"]=[]
hyperParams["diversityFactor"].append(None)
hyperParams["diversityFactor"].append("type:multipleLocations,noLocations:2")
hyperParams["diversityFactor"].append("type:multipleLocations,noLocations:3")
hyperParams["diversityFactor"].append("type:locationVariance,locationVariance:2")
hyperParams["diversityFactor"].append("type:locationVariance,locationVariance:5")
hyperParams["diversityFactor"].append("type:locationVariance,locationVariance:10")

print("echo \"START OF SHELL SCRIPT\"")
print("cd /vulcanscratch/gihan/trojan/")
print("conda activate keras")


if EXP=="a":

    experimentIdx = 0
    for a in hyperParams["poisonSampleCount"]:
        for b in hyperParams["dataset"]:
            for c in hyperParams["experimentType"]:
                prefix = "logs/log-exp-20221214a-{}".format(experimentIdx)


                s = '''python create-trojan.py\\
                        --loggerPrefix {}\\
                        --poisonSampleCount {}\\
                        --dataset {}\\
                        --experimentType {}\\
                        --modelSaveFile {}-savedModel.h5;
                '''.format(prefix, a, b ,c, prefix)

                print(s)

                experimentIdx +=1
    
elif EXP=="b":
    hyperParams["poisonSampleCount"]=[5000,10000]

    experimentIdx = 0
    for a in hyperParams["poisonSampleCount"]:
        for b in hyperParams["dataset"]:
            for c in hyperParams["diversityFactor"]:
                prefix = "logs/b/log-exp-20221214b-{}".format(experimentIdx)


                s = '''python create-trojan.py\\
                        --loggerPrefix {}\\
                        --poisonSampleCount {}\\
                        --dataset {}\\
                        --diversityFactor {}\\
                        --modelSaveFile {}-savedModel.h5;
                '''.format(prefix, a, b ,c, prefix)

                print(s)

                experimentIdx +=1    
            

print("echo \"END OF SHELL SCRIPT\"")


# In[ ]:




