
the csv file is from preprocessing script.
loaders:
  csvfname:
/home/heejong/server/sablab/biopsy-prediction/prostateX-demo-csPCa-prediction-multimodal-sitk-newidx.csv

change checkpoint dir : checkpoint_dir



from csv import reader
import yaml

all_yaml = glob.glob(os.path.join(dirname, 'config/*/*.yaml'))
with open(names, 'r+') as f:
    doc = yaml.load(f)
    orig_dir = doc['trainer']['checkpoint_dir']
    doc['trainer']['checkpoint_dir'] = orig_dir.replace('/home/heejong/projects/biopsy-prediction-result-cnn3d8conv4mp3fcf4-earlystopping-eval/','./checkpoint/')
    yaml.dump(doc, f)



with open(names, 'w+') as out:
    yaml.dump(doc, out)

values = doc['components']['star']['init'][0]['values']
with open('params.csv') as f:
    for i, record in enumerate(reader(f)):
        values['logg'] = record[6]
        with open(f'config-{i}.yaml', 'w') as out:
            yaml.dump(doc, out)
