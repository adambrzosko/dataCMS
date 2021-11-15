#!pip install --user vector
import uproot3
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import xgboost as xgb
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

#~~ Fin edit ~~#
import vector
import awkward as ak  
import numba as nb

treeGG_tt = uproot3.open("/vols/cms/fjo18/Masters2021/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
treeGG_mt = uproot3.open("/vols/cms/fjo18/Masters2021/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_mt_2018.root")["ntuple"]
treeVBF_tt = uproot3.open("/vols/cms/fjo18/Masters2021/MVAFILE_VBFHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
treeVBF_mt = uproot3.open("/vols/cms/fjo18/Masters2021/MVAFILE_VBFHToTauTauUncorrelatedDecay_Filtered_mt_2018.root")["ntuple"]

#~~ Variables to use ~~#
# Need mass of rho and pi0
# E_gamma/E_tauvis for leading photon

# TO ACTUALLY INCLUDE
# Generator Level:
    # tauflag_1, tauflag_2 (for leading subleading)
    # 
# Visible tau 4-momentum

variables_tt_1 = ["tauFlag_1",
                # Generator-level properties, actual decay mode of taus for training
                "gam3_E_1", "gam4_E_1",
                # photon energies (actually only need gam1 for _1 and _2)
                "pi_px_1", "pi_py_1", "pi_pz_1", "pi_E_1", 
                # 4-momenta of the charged pions
                "pi0_px_1", "pi0_py_1", "pi0_pz_1", "pi0_E_1", 
                # 4-momenta of neutral pions
                "gam1_px_1", "gam1_py_1", "gam1_pz_1", "gam1_E_1",
                "gam2_px_1", "gam2_py_1", "gam2_pz_1", "gam2_E_1",
                # 4-momenta of two leading photons
                "tau_px_1", "tau_py_1", "tau_pz_1", "tau_E_1",
                # 4-momenta of 'visible' tau
                "tau_decay_mode_1",
                # HPS algorithm decay mode
                "pt_1",
               ]

variables_tt_2 = ["tauFlag_2", 
                # Generator-level properties, actual decay mode of taus for training
                "gam3_E_2", "gam4_E_2", 
                # photon energies (actually only need gam1 for _1 and _2)
                "pi_px_2", "pi_py_2", "pi_pz_2", "pi_E_2", 
                # 4-momenta of the charged pions
                "pi0_px_2", "pi0_py_2", "pi0_pz_2", "pi0_E_2", 
                # 4-momenta of neutral pions
                "gam1_px_2", "gam1_py_2", "gam1_pz_2", "gam1_E_2",
                "gam2_px_2", "gam2_py_2", "gam2_pz_2", "gam2_E_2",
                # 4-momenta of two leading photons
                "tau_px_2", "tau_py_2", "tau_pz_2", "tau_E_2",
                # 4-momenta of 'visible' tau
                "tau_decay_mode_2", 
                # HPS algorithm decay mode
                "pt_2",
               ]

# List labels for later:
pi_1_4mom = ["pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", ]
pi_2_4mom = ["pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", ]
pi0_1_4mom = ["pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1", ]
pi0_2_4mom = ["pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2", ]
gam1_1_4mom = ["gam1_E_1", "gam1_px_1", "gam1_py_1", "gam1_pz_1", ]
gam2_1_4mom = ["gam2_E_1", "gam2_px_1", "gam2_py_1", "gam2_pz_1", ]
gam1_2_4mom = ["gam1_E_2", "gam1_px_2", "gam1_py_2", "gam1_pz_2", ]
gam2_2_4mom = ["gam2_E_2", "gam2_px_2", "gam2_py_2", "gam2_pz_2", ]
tau_1_4mom = ["tau_E_1", "tau_px_1", "tau_py_1", "tau_pz_1", ]
tau_2_4mom = ["tau_E_2", "tau_px_2", "tau_py_2", "tau_pz_2", ]

dfVBF_tt_1 = treeVBF_tt.pandas.df(variables_tt_1)
dfGG_tt_1 = treeGG_tt.pandas.df(variables_tt_1)
dfVBF_tt_2 = treeVBF_tt.pandas.df(variables_tt_2)
dfGG_tt_2 = treeGG_tt.pandas.df(variables_tt_2)

df_1 = pd.concat([dfVBF_tt_1,dfGG_tt_1], ignore_index=True) 
df_2 = pd.concat([dfVBF_tt_2,dfGG_tt_2], ignore_index=True) 
#combine gluon and vbf data for hadronic modes
del dfVBF_tt_1, dfVBF_tt_2, dfGG_tt_2

df_1.set_axis(variables_tt_2, axis=1, inplace=True) 
# rename axes to the same as variables 2
df_full = pd.concat([df_1, df_2], ignore_index = True)
del df_1, df_2

# rename the axes for the _1 data, so that can concatenate into one tau per row

#~~ Filter decay modes and add in order ~~#
df_DM0 = df_full[
      (df_full["tauFlag_2"] == 0)
]
lenDM0 = df_DM0.shape[0]

df_DM1 = df_full[
      (df_full["tauFlag_2"] == 1)
]
lenDM1 = df_DM1.shape[0]

df_DM2 = df_full[
      (df_full["tauFlag_2"] == 2)
]
lenDM2 = df_DM2.shape[0]

df_DMminus1 = df_full[
      (df_full["tauFlag_2"] == -1)
]
lenDMminus1 = df_DMminus1.shape[0]

df_DM10 = df_full[
      (df_full["tauFlag_2"] == 10)
]
lenDM10 = df_DM10.shape[0]

df_DM11 = df_full[
      (df_full["tauFlag_2"] == 11)
]
lenDM11 = df_DM11.shape[0]

df_ordered = pd.concat([df_DM0, df_DM1, df_DM2, df_DMminus1, df_DM10, df_DM11], ignore_index = True)
del df_DM0, df_DM1, df_DM2, df_DMminus1, df_DM10, df_DM11, df_full

# Creating target labels 
d_DM0 = pd.DataFrame({'col': np.zeros(lenDM0)})
d_DM1 = pd.DataFrame({'col': np.ones(lenDM1)})
d_DM2 = pd.DataFrame({'col': 2 * np.ones(lenDM2)})
d_DMminus1 = pd.DataFrame({'col': 3 * np.ones(lenDMminus1)})
d_DM10 = pd.DataFrame({'col': 3 * np.ones(lenDM10)})
d_DM11 = pd.DataFrame({'col': 3 * np.ones(lenDM11)})

y = pd.concat([d_DM0, d_DM1, d_DM2, d_DMminus1, d_DM10, d_DM11], ignore_index = True)

del d_DM0, d_DM1, d_DM2, d_DMminus1, d_DM10, d_DM11

def inv_mass(Energ,Px,Py,Pz):
    vect = vector.obj(px=Px, py=Py, pz=Pz, E=Energ)
    return vect.mass

df_ordered["pi0_2mass"] = inv_mass(df_ordered["pi0_E_2"],df_ordered["pi0_px_2"],df_ordered["pi0_py_2"],df_ordered["pi0_pz_2"]) #pion masses

def rho_mass(dataframe, momvariablenames_1, momvariablenames_2):
    momvect1 = vector.obj(px = dataframe[momvariablenames_1[1]],\
                       py = dataframe[momvariablenames_1[2]],\
                       pz = dataframe[momvariablenames_1[3]],\
                       E = dataframe[momvariablenames_1[0]])
    momvect2 = vector.obj(px = dataframe[momvariablenames_2[1]],\
                       py = dataframe[momvariablenames_2[2]],\
                       pz = dataframe[momvariablenames_2[3]],\
                       E = dataframe[momvariablenames_2[0]])
    rho_vect = momvect1+momvect2
    dataframe["rho_mass"] = inv_mass(rho_vect.E, rho_vect.px, rho_vect.py, rho_vect.pz) #rho masses
    
rho_mass(df_ordered, pi_2_4mom, pi0_2_4mom)
# rho mass is the addition of the four-momenta of the charged and neutral pions

df_ordered["E_gam/E_tau"] = df_ordered["gam1_E_2"].divide(df_ordered["tau_E_2"]) #Egamma/Etau
df_ordered["E_pi/E_tau"] = df_ordered["pi_E_2"].divide(df_ordered["tau_E_2"]) #Epi/Etau
df_ordered["E_pi0/E_tau"] = df_ordered["pi0_E_2"].divide(df_ordered["tau_E_2"]) #Epi0/Etau

def pi0_mass(dataframe, momvariablenames_1, momvariablenames_2):
    momvect1 = vector.obj(px = dataframe[momvariablenames_1[1]],\
                       py = dataframe[momvariablenames_1[2]],\
                       pz = dataframe[momvariablenames_1[3]],\
                       E = dataframe[momvariablenames_1[0]])
    momvect2 = vector.obj(px = dataframe[momvariablenames_2[1]],\
                       py = dataframe[momvariablenames_2[2]],\
                       pz = dataframe[momvariablenames_2[3]],\
                       E = dataframe[momvariablenames_2[0]])
    pi0_vect = momvect1+momvect2
    dataframe["Mpi0"] = inv_mass(pi0_vect.E, pi0_vect.px, pi0_vect.py, pi0_vect.pz) #rho masses

pi0_mass(df_ordered, gam1_2_4mom, gam2_2_4mom)

def tau_eta(dataframe, momvariablenames_1):
    momvect1 = vector.obj(px = dataframe[momvariablenames_1[1]],\
                       py = dataframe[momvariablenames_1[2]],\
                       pz = dataframe[momvariablenames_1[3]],\
                       E = dataframe[momvariablenames_1[0]])
    dataframe["tau_eta"] = momvect1.eta  #tau eta (tau pt just a variable)
    
tau_eta(df_ordered, tau_2_4mom)

def ang_var(dataframe, momvariablenames_1, momvariablenames_2): #same for gammas and pions
    momvect1 = vector.obj(px = dataframe[momvariablenames_1[1]],\
                       py = dataframe[momvariablenames_1[2]],\
                       pz = dataframe[momvariablenames_1[3]],\
                       E = dataframe[momvariablenames_1[0]])
    momvect2 = vector.obj(px = dataframe[momvariablenames_2[1]],\
                       py = dataframe[momvariablenames_2[2]],\
                       pz = dataframe[momvariablenames_2[3]],\
                       E = dataframe[momvariablenames_2[0]])
    
    diffphi = momvect1.phi - momvect2.phi
    diffeta = momvect1.eta - momvect2.eta
    diffr = np.sqrt(diffphi**2 + diffeta**2)
    Esum = dataframe[momvariablenames_1[0]] + dataframe[momvariablenames_2[0]]
    return diffphi, diffeta, diffr, diffphi/Esum, diffeta/Esum, diffr/Esum

df_ordered["delR_gam"],df_ordered["delPhi_gam"],df_ordered["delEta_gam"],df_ordered["delR_xE_gam"],df_ordered["delPhi_xE_gam"],\
df_ordered["delEta_xE_gam"] = ang_var(df_ordered, gam1_2_4mom, gam2_2_4mom)
df_ordered["delR_pi"],df_ordered["delPhi_pi"],df_ordered["delEta_pi"],df_ordered["delR_xE_pi"],df_ordered["delPhi_xE_pi"],\
df_ordered["delEta_xE_pi"] = ang_var(df_ordered, pi0_2_4mom, pi_2_4mom)


def R2(dataframe, variables, number):
    a = []
    tau = [tau_1_4mom,tau_2_4mom]
    for i in variables:
        a.append((ang_var(dataframe,tau[number-1],i)[2]**2)*(dataframe["pt_"+str(number)]**2))
    return sum(a)/sum(dataframe["pt_"+str(number)]**2)    

df_ordered["R2"] = R2(df_ordered, [pi_2_4mom, pi0_2_4mom, gam1_2_4mom, gam2_2_4mom], 2)

X = df_ordered.drop(["tauFlag_2", 
                # Generator-level properties, actual decay mode of taus for training
                "gam3_E_2", "gam4_E_2", 
                # photon energies (actually only need gam1 for _1 and _2)
                "pi_px_2", "pi_py_2", "pi_pz_2", 
                # 4-momenta of the charged pions
                "pi0_px_2", "pi0_py_2", "pi0_pz_2", 
                # 4-momenta of neutral pions
                "gam1_px_2", "gam1_py_2", "gam1_pz_2", "gam1_E_2",
                "gam2_px_2", "gam2_py_2", "gam2_pz_2", "gam2_E_2",
                # 4-momenta of two leading photons
                "tau_px_2", "tau_py_2", "tau_pz_2", "tau_E_2",
                # 4-momenta of 'visible' tau
           ], axis=1).reset_index(drop=True)
del df_ordered

y1= y.values.ravel()
X1_train, X1_test, y1_train, y1_test  = train_test_split(
    X,
    y1,
    test_size=0.5,
    random_state=123456,
    stratify = y1
)

xgb_params = {
    "objective": "multi:softprob",
    "max_depth": 5,
    "learning_rate": 0.05,
    "silent": 1,
    "n_estimators": 2000,
    "subsample": 0.9,
    "seed": 123451, 
    "num_class": 4,
}
xgb_clf = xgb.XGBClassifier(**xgb_params)
xgb_clf.fit(X1_train,
            y1_train,
    early_stopping_rounds=100, # stops the training if doesn't improve after 200 iterations
    eval_set=[(X1_train, y1_train), (X1_test, y1_test)],
    eval_metric = "merror", # can use others
    verbose=True,
)

y1_proba = xgb_clf.predict_proba(X1_test)

import pickle
file_name = "xgb_clf_model2.pkl"

# save
pickle.dump(xgb_clf, open(file_name, "wb"))
xgb_clf.save_model('xgb_clf_model2.json')

y1_proba_round = np.zeros(y1_test.shape)
for a in range(len(y1_proba)):
    y1_proba_round[a] = list(y1_proba[a]).index(max(y1_proba[a]))
    
truelabels = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]) #for true modes 0,1,2,3 (other)
lengthstrue = [0,0,0,0]
lengthspred = [0,0,0,0]
for a in range(len(y1_test)):
    truelabels[int(y1_test[a])][int(y1_proba_round[a])] +=1
    lengthstrue[int(y1_test[a])] +=1
    lengthspred[int(y1_proba_round[a])] +=1
truelabelpurity = truelabels/lengthspred
truelabelefficiency = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype = float)
for a in range(4):
    for b in range(4):
        truelabelefficiency[a][b] = truelabels[a][b]/lengthstrue[a]
        
#~~ PLOTTING CONFUSION MATRICES ~~#
plt.rcParams.update({'figure.autolayout': True})
labellist = [r'$\pi^{\pm}$', r'$\pi^{\pm} \pi^0$', r'$\pi^{\pm} 2\pi^0$', 'Other']
fig, ax = plt.subplots(1,2)
plt.tight_layout()

ax[0].imshow(truelabelefficiency, cmap = 'Blues')
for i in range(4):
    for j in range(4):
        if truelabelefficiency[i, j] > 0.5:
            text = ax[0].text(j, i, round(truelabelefficiency[i, j], 3),
                           ha="center", va="center", color="w")
        else:
            text = ax[0].text(j, i, round(truelabelefficiency[i, j], 3),
                           ha="center", va="center", color="black")

        
ax[0].set_title('Efficiency')
ax[0].set_xticks([0,1,2,3])
ax[0].set_yticks([0,1,2,3])
ax[0].set_xticklabels(labellist)
ax[0].set_yticklabels(labellist)
ax[0].set_xlabel('Predicted Mode')
ax[0].set_ylabel('True Mode')


ax[1].imshow(truelabelpurity, cmap = 'Blues')
for i in range(4):
    for j in range(4):
        if truelabelpurity[i, j] > 0.5:
            text = ax[1].text(j, i, round(truelabelpurity[i, j], 3),
                           ha="center", va="center", color="w")
        else:
            text = ax[1].text(j, i, round(truelabelpurity[i, j], 3),
                           ha="center", va="center", color="black")

ax[1].set_title('Purity')
ax[1].set_xticks([0,1,2,3])
ax[1].set_yticks([0,1,2,3])
ax[1].set_xticklabels(labellist)
ax[1].set_yticklabels(labellist)
ax[1].set_xlabel('Predicted Mode')
ax[1].set_ylabel('True Mode')


plt.savefig('EffPureConfMatrix_2000gen.png', dpi = 500)
