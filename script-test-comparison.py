import pandas as pd
import numpy as np
import scipy.stats
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix, accuracy_score
import sklearn.linear_model
import numpy
from scipy.stats import bootstrap
from roc_comparison import compare_auc_delong_xu


def accuracy_with_threshold(target, prediction):
    fpr, tpr, thresholds = roc_curve(target, prediction)
    idx = np.argmax(tpr - fpr)
    return accuracy_score(target, prediction > thresholds[idx]), thresholds[idx]

def specificity_sensitivity(target, prediction):
    tn, fp, fn, tp = confusion_matrix(target, prediction).ravel()
    sensitivity = tp / (tp + fn)  # overall positives
    specificity = tn / (tn + fp)  # overall negatives
    return specificity, sensitivity

def get_auc(target, pred):
    return roc_auc_score(target, pred)

def print_all_results(target, prediction, n_trials, rng, bootstrap_subsample=None):
    acc, accthr = accuracy_with_threshold(target, prediction)
    auc = get_auc(target, prediction)
    prediction_binary = prediction > accthr
    specificity, sensitivity = specificity_sensitivity(target, prediction_binary) # requires binary

    if bootstrap_subsample == None:
        resauc = bootstrap((target, prediction), get_auc, vectorized=False, paired=True,
                        n_resamples=n_trials, random_state=rng) # requires binary

        resacc = bootstrap((target, prediction_binary), get_accuracy, vectorized=False, paired=True,
                        n_resamples=n_trials, random_state=rng) # requires binary

        resspe = bootstrap((target, prediction_binary), get_specificity, vectorized=False, paired=True,
                        n_resamples=n_trials, random_state=rng) # requires binary

        ressen = bootstrap((target, prediction_binary), get_sensitivity, vectorized=False, paired=True,
                        n_resamples=n_trials, random_state=rng) # requires binary

        print(f'AUC: {auc:.2f} CI: {resauc.confidence_interval.low:.2f} / {resauc.confidence_interval.high:.2f} \n '
              f'Acc: {acc:.2f} CI: {resacc.confidence_interval.low:.2f} / {resacc.confidence_interval.high:.2f} \n '
              f'Sen: {sensitivity:.2f} CI: {ressen.confidence_interval.low:.2f} / {ressen.confidence_interval.high:.2f} \n '
              f'Spe: {specificity:.2f} CI: {resspe.confidence_interval.low:.2f} / {resspe.confidence_interval.high:.2f} \n ')

    else:

        n_bootstraps = n_trials
        bootstrapped_scores = []

        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(target), bootstrap_subsample) # low high size
            if len(np.unique(target[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue

            score = roc_auc_score(target[indices], prediction[indices])
            bootstrapped_scores.append(score)
            # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

        # import matplotlib.pyplot as plt
        # plt.hist(bootstrapped_scores, bins=50)
        # plt.title('Histogram of the bootstrapped ROC AUC scores')
        # plt.show()

        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()

        # Computing the lower and upper bound of the 90% confidence interval
        # You can change the bounds percentiles to 0.025 and 0.975 to get
        # a 95% confidence interval instead.
        confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
        print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
            confidence_lower, confidence_upper))

    return accthr

def get_specificity(target, pred):
    tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
    sensitivity = tp / (tp + fn)  # overall positives
    specificity = tn / (tn + fp)  # overall negatives
    return specificity

def get_sensitivity(target, pred):
    tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
    sensitivity = tp / (tp + fn)  # overall positives
    specificity = tn / (tn + fp)  # overall negatives
    return sensitivity

def get_accuracy(target, pred):
    return accuracy_score(target, pred)


target = np.array(pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-t2-15init.csv')['ClinSig-target'])
prediction_t2 = np.array(pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-t2-15init.csv')['ClinSig-ensemble'])
prediction_adc = np.array(pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-adc-15init.csv')['ClinSig-ensemble'])
prediction_dwib800 = np.array(pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-dwib800-15init.csv')['ClinSig-ensemble'])
prediction_ktrans = np.array(pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-ktrans-15init.csv')['ClinSig-ensemble'])
prediction_1c_t2_adc_dwib800 = (prediction_t2 + prediction_adc + prediction_dwib800)/3
prediction_1c_t2_adc_ktrans = (prediction_t2 + prediction_adc + prediction_ktrans)/3
prediction_3c_t2_adc_dwib800 = np.array(pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-3channel-t2-adc-dwib800.csv')['ClinSig-ensemble'])
prediction_3c_t2_adc_ktrans = np.array(pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-3channel-t2-adc-ktrans.csv')['ClinSig-ensemble'])


t2_5 =  np.array(pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-t2.csv')['ClinSig-ensemble'])
adc_5 =  np.array(pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-adc.csv')['ClinSig-ensemble'])
ktrans_5 = np.array(pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-ktrans.csv')['ClinSig-ensemble'])
dwi_5 = np.array(pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-dwib800.csv')['ClinSig-ensemble'])

prediction_1c_t2_adc_dwib800 = (t2_5 + adc_5 + dwi_5)/3
prediction_1c_t2_adc_ktrans = (t2_5 + adc_5 + ktrans_5)/3

### TODO: additional combinations
def get_n_init_avg(result, n_init):
    prediction = np.zeros(len(result))
    for n in range(n_init):
        prediction += result[f'CilnSig-seed{n}']
    return prediction / n_init

t2result = pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-t2-15init.csv')
adcresult = result = pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-adc-15init.csv')
dwiresult = result = pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-dwib800-15init.csv')
ktransresult = result = pd.read_csv('./_prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-ktrans-15init.csv')
rng_seed = 42  # control reproducibility
rng = np.random.RandomState(rng_seed)
# rng = np.random.default_rng()
n_trials = 10000
print('T2-ADC:')
print_all_results(target, (get_n_init_avg(t2result, 8)+get_n_init_avg(adcresult, 8))/2, n_trials, rng)
print('T2-DWI:')
print_all_results(target, (get_n_init_avg(t2result, 8)+get_n_init_avg(dwiresult, 8))/2, n_trials, rng)
print('T2-ktrans:')
print_all_results(target, (get_n_init_avg(t2result, 8)+get_n_init_avg(ktransresult, 8))/2, n_trials, rng)
print('ADC-DWI:')
print_all_results(target, (get_n_init_avg(adcresult, 8)+get_n_init_avg(dwiresult, 8))/2, n_trials, rng)
print('ADC-ktrans:')
print_all_results(target, (get_n_init_avg(adcresult, 8)+get_n_init_avg(ktransresult, 8))/2, n_trials, rng)
print('ktrans-DWI:')
print_all_results(target, (get_n_init_avg(ktransresult, 8)+get_n_init_avg(dwiresult, 8))/2, n_trials, rng)

print('T2-ADC-DWI:')
print_all_results(target,
                  (get_n_init_avg(t2result, 5)+get_n_init_avg(adcresult, 5)+get_n_init_avg(dwiresult, 5))/3,
                  n_trials, rng)
print('ADC-DWI-ktrans:')
print_all_results(target,
                  (get_n_init_avg(adcresult, 5)+get_n_init_avg(dwiresult, 5)+get_n_init_avg(ktransresult, 5))/3,
                  n_trials, rng)
print('T2-ktrans-DWI:')
print_all_results(target,
                  (get_n_init_avg(t2result, 5)+get_n_init_avg(ktransresult, 5)+get_n_init_avg(dwiresult, 5))/3,
                  n_trials, rng)
print('T2-ADC-ktrans:')
print_all_results(target,
                  (get_n_init_avg(t2result, 5)+get_n_init_avg(adcresult, 5)+get_n_init_avg(ktransresult, 5))/3,
                  n_trials, rng)

print('T2-ADC-DWI-ktrans:')
print_all_results(target,
                  (get_n_init_avg(t2result, 4)+get_n_init_avg(adcresult, 4)+\
                   get_n_init_avg(ktransresult, 4)+get_n_init_avg(dwiresult, 4))/4,
                  n_trials, rng)


print('t2', get_auc(target, prediction_t2))
print('adc', get_auc(target, prediction_adc))
print('dwi', get_auc(target, prediction_dwib800))
print('ktrans', get_auc(target, prediction_ktrans))
print('1c t2 adc dwi', get_auc(target, prediction_1c_t2_adc_dwib800))
print('1c t2 adc ktrans', get_auc(target, prediction_1c_t2_adc_ktrans))
print('1c t2 adc dwi ktrans', get_auc(target, (t2_5+adc_5+dwi_5+ktrans_5)/4)) # not fair though
print('3c w dwi', get_auc(target, prediction_3c_t2_adc_dwib800))
print('3c w ktrans', get_auc(target, prediction_3c_t2_adc_ktrans))

# statistically significant difference ?

rng_seed = 42  # control reproducibility
rng = np.random.RandomState(rng_seed)
# rng = np.random.default_rng()
n_trials = 10000

print_all_results(target, prediction_t2, n_trials, rng)
print_all_results(target, prediction_1c_t2_adc_dwib800, n_trials, rng)


x_distr = scipy.stats.norm(0.5, 1)
y_distr = scipy.stats.norm(-0.5, 1)
sample_size_x = 7
sample_size_y = 14
n_trials = 1000
aucs = numpy.empty(n_trials)
variances = numpy.empty(n_trials)
numpy.random.seed(1234235)
labels = numpy.concatenate([numpy.ones(sample_size_x), numpy.zeros(sample_size_y)])
for trial in range(n_trials):
    scores = numpy.concatenate([
        x_distr.rvs(sample_size_x),
        y_distr.rvs(sample_size_y)])
    aucs[trial] = sklearn.metrics.roc_auc_score(labels, scores)
    auc_delong, variances[trial] = compare_auc_delong_xu.delong_roc_variance(
        labels, scores)

print(f"Experimental variance {variances.mean():.4f}, "
      f"computed vairance {aucs.var():.4f}, {n_trials} trials")


def delong_pvalue(target, prediction1, prediction2):
    return 10**(compare_auc_delong_xu.delong_roc_test(target, prediction1, prediction2))

print('t2 / fusion 3c dwi', delong_pvalue(target, prediction_t2, prediction_1c_t2_adc_dwib800))
print('t2 / fusion 3c ktrans', delong_pvalue(target, prediction_t2, prediction_1c_t2_adc_ktrans))

print('adc / fusion 3c dwi', delong_pvalue(target, prediction_adc, prediction_1c_t2_adc_dwib800))
print('adc / fusion 3c ktrans', delong_pvalue(target, prediction_adc, prediction_1c_t2_adc_ktrans))

print('dwi / fusion 3c dwi', delong_pvalue(target, prediction_dwib800, prediction_1c_t2_adc_dwib800))
print('dwi / fusion 3c ktrans', delong_pvalue(target, prediction_dwib800, prediction_1c_t2_adc_ktrans))

print('ktrans / fusion 3c dwi', delong_pvalue(target, prediction_ktrans, prediction_1c_t2_adc_dwib800))
print('ktrans / fusion 3c ktrans', delong_pvalue(target, prediction_ktrans, prediction_1c_t2_adc_ktrans))

###
## TODO: DeLong add results on plot (* 0.05 / ** 0.005)
# t2 / fusion 3c dwi [[0.02504358]] *
# t2 / fusion 3c ktrans [[0.08377477]]
# adc / fusion 3c dwi [[0.14612564]]
# adc / fusion 3c ktrans [[0.25510883]]
# dwi / fusion 3c dwi [[0.07023753]]
# dwi / fusion 3c ktrans [[0.21914456]]
# ktrans / fusion 3c dwi [[0.01085641]] *
# ktrans / fusion 3c ktrans [[0.00093912]] **


## TODO McNemar (nothing related to the true accuracy but finding if there is a difference)
acc, thr_t2 = accuracy_with_threshold(target, prediction_t2)
acc, thr_adc = accuracy_with_threshold(target, prediction_adc)
acc, thr_dwib800 = accuracy_with_threshold(target, prediction_dwib800)
acc, thr_ktrans = accuracy_with_threshold(target, prediction_ktrans)
acc, thr_1c_t2_adc_dwib800 = accuracy_with_threshold(target, prediction_1c_t2_adc_dwib800)
acc, thr_1c_t2_adc_ktrans = accuracy_with_threshold(target, prediction_1c_t2_adc_ktrans)
acc, thr_3c_t2_adc_dwib800 = accuracy_with_threshold(target, prediction_3c_t2_adc_dwib800)
acc, thr_3c_t2_adc_ktrans = accuracy_with_threshold(target, prediction_3c_t2_adc_ktrans)

from statsmodels.stats.contingency_tables import mcnemar
def mcnemar_stat(prediction1, thr1, prediction2, thr2):
    data = confusion_matrix(prediction1>=thr1, prediction2>=thr2)
    # print(data)
    return print(mcnemar(data, exact=True, correction=False))

print('t2 / fusion dwi')
mcnemar_stat(prediction_t2, thr_t2, prediction_1c_t2_adc_dwib800, thr_1c_t2_adc_dwib800)
print('adc / fusion dwi')
mcnemar_stat(prediction_adc, thr_adc, prediction_1c_t2_adc_dwib800, thr_1c_t2_adc_dwib800)
print('dwi / fusion dwi')
mcnemar_stat(prediction_dwib800, thr_dwib800, prediction_1c_t2_adc_dwib800, thr_1c_t2_adc_dwib800)
print('ktrans / fusion dwi')
mcnemar_stat(prediction_ktrans, thr_ktrans, prediction_1c_t2_adc_dwib800, thr_1c_t2_adc_dwib800)

print('t2 / fusion ktrans')
mcnemar_stat(prediction_t2, thr_t2, prediction_1c_t2_adc_dwib800, thr_1c_t2_adc_ktrans)
print('adc / fusion ktrans')
mcnemar_stat(prediction_adc, thr_adc, prediction_1c_t2_adc_dwib800, thr_1c_t2_adc_ktrans)
print('dwi / fusion ktrans')
mcnemar_stat(prediction_dwib800, thr_dwib800, prediction_1c_t2_adc_dwib800, thr_1c_t2_adc_ktrans)
print('ktrans / fusion ktrans')
mcnemar_stat(prediction_ktrans, thr_ktrans, prediction_1c_t2_adc_dwib800, thr_1c_t2_adc_ktrans)


## TODO different combinations of pairs & auc

