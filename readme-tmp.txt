conda env create -f environment.yaml


the csv file is from preprocessing script.
loaders:
  csvfname:
/home/heejong/server/sablab/biopsy-prediction/prostateX-demo-csPCa-prediction-multimodal-sitk-newidx.csv

change checkpoint dir : checkpoint_dir



import yaml


all_yaml = glob.glob(os.path.join(dirname, 'config/*/*.yaml'))

for names in all_yaml:
    f = open(names, 'r')
    newf = f.read().replace('/home/heejong/projects/biopsy-prediction-result-cnn3d8conv4mp3fcf4-earlystopping-eval/augment2', './checkpoint/augment')
    f.close()

    with open(names, 'w') as ff:
        ff.write(newf)

## for the server change the filename
''
for names in all_yaml:
    f = open(names, 'r')
    newf = f.read().replace('/home/heejong/data/prostatex/demo-h5-sitk/prostateX-demo-csPCa-prediction-multimodal-sitk-newidx.csv',
                '/share/sablab/nfs04/data/PROSTATEx/preprocessed/PROSTATEx-new/prostateX-demo-csPCa-prediction-multimodal-sitk-newidx-scratch.csv')
    f.close()

    with open(names, 'w') as ff:
        ff.write(newf)


 -> sablab-gpu-12: 8 a6000 [cpu: 16/32, gres/gpu: 4/8, mem: 192 GB/385 GB] []

# make yaml
# change name
# run on server
