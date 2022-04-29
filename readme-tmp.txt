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

## for back to local filename
''
for names in all_yaml:
    f = open(names, 'r')
    newf = f.read().replace('/share/sablab/nfs04/data/PROSTATEx/preprocessed/PROSTATEx-new/prostateX-demo-csPCa-prediction-multimodal-sitk-newidx-scratch.csv',
    '/home/heejong/data/prostatex/demo-h5-sitk/prostateX-demo-csPCa-prediction-multimodal-sitk-newidx.csv')
    f.close()

    with open(names, 'w') as ff:
        ff.write(newf)

 -> sablab-gpu-12: 8 a6000 [cpu: 16/32, gres/gpu: 4/8, mem: 192 GB/385 GB] []

# make yaml
# change name
# run on server

python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc.yaml& python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed1.yaml& python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed2.yaml;python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed3.yaml& python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed4.yaml& python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed5.yaml;python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed6.yaml& python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed7.yaml& python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed8.yaml;python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed9.yaml& python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed10.yaml& python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed11.yaml;python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed12.yaml& python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed13.yaml& python -u ./train.py ./config/adc/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed14.yaml;\
python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc.yaml& python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed1.yaml& python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed2.yaml;python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed3.yaml& python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed4.yaml& python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed5.yaml;python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed6.yaml& python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed7.yaml& python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed8.yaml;python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed9.yaml& python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed10.yaml& python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed11.yaml;python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed12.yaml& python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed13.yaml& python -u ./train.py ./config/dwib800/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed14.yaml;\
python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc.yaml& python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed1.yaml& python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed2.yaml;python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed3.yaml& python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed4.yaml& python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed5.yaml;python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed6.yaml& python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed7.yaml& python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed8.yaml;python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed9.yaml& python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed10.yaml& python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed11.yaml;python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed12.yaml& python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed13.yaml& python -u ./train.py ./config/ktrans/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed14.yaml;\
python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc.yaml& python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed1.yaml& python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed2.yaml;python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed3.yaml& python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed4.yaml& python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed5.yaml;python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed6.yaml& python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed7.yaml& python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed8.yaml;python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed9.yaml& python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed10.yaml& python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed11.yaml;python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed12.yaml& python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed13.yaml& python -u ./train.py ./config/t2/augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed14.yaml;\




python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc.yaml& python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed1.yaml& python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed2.yaml;python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed3.yaml& python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed4.yaml& python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed5.yaml;python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed6.yaml& python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed7.yaml& python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed8.yaml;python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed9.yaml& python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed10.yaml& python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed11.yaml;python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed12.yaml& python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed13.yaml& python -u ./train.py ./config/t2-adc-ktrans/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed14.yaml;\

python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc.yaml& python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed1.yaml& python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed2.yaml;python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed3.yaml& python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed4.yaml& python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed5.yaml;python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed6.yaml& python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed7.yaml& python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed8.yaml;python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed9.yaml& python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed10.yaml& python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed11.yaml;python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed12.yaml& python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed13.yaml& python -u ./train.py ./config/t2-adc-dwib800/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed14.yaml;\


#  -> sablab-gpu-12: 8 a6000 [cpu: 16/32, gres/gpu: 4/8, mem: 192 GB/385 GB] []
# -> sablab-gpu-08: 8 2080ti [cpu: 5/24, gres/gpu: 5/8, mem: 40 GB/257 GB] []
# -> sablab-gpu-03: 4 1080ti [cpu: 4/16, gres/gpu: 2/4, mem: 62 GB/128 GB] []

sbatch --requeue -p sablab-highprio -w sablab-gpu-12 -t 48:00:00 --gres=gpu:4 --cpus-per-task=16 --mem=180G -e ./joblog/3cktrans.err -o ./joblog/3cktrans.log ./sbatch-ktrans.sh
sbatch --requeue -p sablab-highprio -w sablab-gpu-08 -t 48:00:00 --gres=gpu:3 --cpus-per-task=19 --mem=210G -e ./joblog/3cdwib800.err -o ./joblog/3cdwib800.log ./sbatch-dwib800.sh
sbatch --requeue -p sablab-highprio -w sablab-gpu-03 -t 48:00:00 --gres=gpu:2 --cpus-per-task=12 --mem=64G -e ./joblog/1cadc.err -o ./joblog/1cadc.log ./sbatch-adc.sh