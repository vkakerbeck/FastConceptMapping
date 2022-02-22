import os
import matplotlib.pyplot as plt
import numpy as np
#from JSAnimation import IPython_display
from matplotlib import animation
import matplotlib.patches as mpatches
import seaborn as sns
#import cv2
#import imageio
#import mpld3
import scipy.misc
from itertools import combinations
from scipy.spatial import distance
import pandas as pd

def loadRecordingsAtPath(path, params, conditionID, numActs, getLabels=False, checkData = False):
    print("Loading " + str(params['train']) + " - " + conditionID + ' from ' + path)
    obs = np.load(path+'visobs.npy', mmap_mode='r')
    data = {"observations":obs[1:]}
   
    data['name'] = params['train']
    
    if params['useCurEnc']:
        enc = np.load(path+'enc_cur_state.npy')
        print('Loaded curious of shape: ' + str(np.array(enc).shape))
    else:
        enc = np.load(path+'encodings.npy')
    enc = enc.reshape(enc.shape[0],enc.shape[-1])
    if params['onlyVisual'] and params['isPPO']:
        enc = enc[:,:int(enc.shape[-1]/2)]
    data["encodings_"+conditionID] = enc[1:]
    
    if params['isPPO']:
        vec = np.load(path+'vecobs.npy')[:,1:]
        val = np.load(path+'values.npy')[1:]
        rew = np.load(path+'rewards.npy')#[1:]    
        act = np.load(path+'actions.npy')[1:]
        #act_combs, act_labels = actionProcessing(act,numActs)
        
        data.update({"vectorObs":vec,
                     "values_"+conditionID:val,
                     "rewards":rew,
                     "actions_"+conditionID:act
                     #"act_combs_"+conditionID:act_combs,
                     #"act_labels_"+conditionID:act_labels
                    })
            
    else:
        print('please provide data from a PPO agent with rewards & actions.')
        
    if params['isCurious']:
        pred_s = np.load(path+'pred_state.npy')
        pred_s = pred_s.reshape(pred_s.shape[0],pred_s.shape[-1])
        if params['onlyVisual'] and params['isPPO']:
            pred_s = pred_s[:,:int(pred_s.shape[-1]/2)]
        pred_a = np.load(path+'pred_act.npy')
        data["pred_states_"+conditionID] = pred_s[1:]
        data["pred_actions_"+conditionID] = pred_a[1:]
        
        if data["values_"+conditionID].shape[0] > data['rewards'].shape[0]:#sometimes one extra frame is recorded in some variables
            runLen = data['rewards'].shape[0]
            print('False Logging in ' + data["name"] + ' - Rewards length: ' + str(data['rewards'].shape[0]) + ' Values length: ' + str(data['values_'+conditionID].shape[0]))
            #data["pred_states_"+conditionID] = data["pred_states_"+conditionID][-runLen:,:]
            #data["pred_actions_"+conditionID] = data["pred_actions_"+conditionID][-runLen:,:,:]
            #data["actions_"+conditionID] = data["actions_"+conditionID][-runLen:,:,:]
            #data["values_"+conditionID] = data["values_"+conditionID][-runLen:,:,:]
            #data["act_combs_"+conditionID] = data["act_combs_"+conditionID][-runLen:]
    
    if getLabels and len(params['labelPath']) > 0:
        labels = pd.read_csv(params['path'] + params['labelPath'])
        data["labels"] = np.array(labels)[:,2:]
    
    if checkData:
        for key in data.keys():
            print(str(key) + ': ' + str(np.array(data[key]).shape))
    
    return data

def LoadData(c1Params, c2Params, c3Params, numActs=5, checkData=False):
    # TODO: add checkData into all loadRecordingsAtPath calls
    c1DataO1 = loadRecordingsAtPath(c1Params['path'] + c1Params['runNumber'] + "_Same/", c1Params, 'C1O1', numActs, getLabels = True)
    c1DataO2 = loadRecordingsAtPath(c2Params['path'] + c2Params['runNumber'] + "_" + c1Params['train'] + "/", c1Params, 'C1O2', numActs)
    c1DataO3 = loadRecordingsAtPath(c3Params['path'] + c3Params['runNumber'] + "_" + c1Params['train'] + "/", c1Params, 'C1O3', numActs)
    
    
    c2DataO1 = loadRecordingsAtPath(c1Params['path'] + c1Params['runNumber'] + "_" + c2Params['train'] + "/", c2Params, 'C2O1', numActs)
    c2DataO2 = loadRecordingsAtPath(c2Params['path'] + c2Params['runNumber'] + "_Same/", c2Params, 'C2O2', numActs, getLabels = True)
    c2DataO3 = loadRecordingsAtPath(c3Params['path'] + c3Params['runNumber'] + "_" + c2Params['train'] + "/", c2Params, 'C2O3', numActs)
    
    
    c3DataO1 = loadRecordingsAtPath(c1Params['path'] + c1Params['runNumber'] + "_" + c3Params['train'] + "/", c3Params, 'C3O1', numActs)
    c3DataO2 = loadRecordingsAtPath(c2Params['path'] + c2Params['runNumber'] + "_" + c3Params['train'] + "/", c3Params, 'C3O2', numActs)
    c3DataO3 = loadRecordingsAtPath(c3Params['path'] + c3Params['runNumber'] + "_Same/", c3Params, 'C3O3', numActs, getLabels = True)
    
    if checkData:
        if (c1DataO1["observations"] == c2DataO1["observations"]).all()  and (c1DataO1["observations"] == c3DataO1["observations"]).all():
            print("all good with observations 1 (" + c1Params['train'] + ')')

        if (c1DataO2["observations"] == c2DataO2["observations"]).all()  and (c1DataO2["observations"] == c3DataO2["observations"]).all():
            print("all good with observations 2 (" + c2Params['train'] + ')')

        if (c1DataO3["observations"] == c2DataO3["observations"]).all()  and (c1DataO3["observations"] == c3DataO3["observations"]).all():
            print("all good with observations 3 (" + c3Params['train'] + ')')
    
    toMerge = ['encodings', 'values', 'actions']#, 'act_combs', 'act_labels']
    # TODO: merge pred_states and pred_actions of agents with curiosity
    
    maxObs = np.min([c1DataO1["observations"].shape[0],c1DataO2["observations"].shape[0],c1DataO3["observations"].shape[0]])
    print('Observation Clip at: ' + str(maxObs))
    # As Observations and rewards are the same for all of them they are only saved once in condition 1 to save memory
    c1Data = {'name': c1Params['train'], 'title': c1Params['title'],
              'observations': np.vstack((c1DataO1["observations"][:maxObs],c2DataO2["observations"][:maxObs],c3DataO3["observations"][:maxObs])),
              'vectorObs': np.hstack((c1DataO1["vectorObs"][:,:maxObs],c2DataO2["vectorObs"][:,:maxObs],c3DataO3["vectorObs"][:,:maxObs])),
             'rewards': np.hstack((c1DataO1["rewards"][:maxObs],c2DataO2["rewards"][:maxObs],c3DataO3["rewards"][:maxObs])),
             'labels': np.vstack((c1DataO1["labels"][:maxObs],c2DataO2["labels"][:maxObs],c3DataO3["labels"][:maxObs]))}
    c2Data = {'name': c2Params['train'], 'title': c2Params['title']}
    c3Data = {'name': c3Params['train'], 'title': c3Params['title']}
    for key in toMerge:
        c1Data[key] = np.vstack((c1DataO1[key + '_C1O1'][:maxObs],c1DataO2[key + '_C1O2'][:maxObs],c1DataO3[key + '_C1O3'][:maxObs]))
        print('Condition ' + c1Params['train'] + ' merged ' + str(key) + ' of shapes: ' + 
              str(np.array(c1DataO1[key + "_C1O1"]).shape) + ', ' + 
              str(np.array(c1DataO2[key + "_C1O2"]).shape) + ', ' + 
              str(np.array(c1DataO3[key + "_C1O3"]).shape) + ' new shape: ' + str(np.array(c1Data[key]).shape))
        c2Data[key] = np.vstack((c2DataO1[key + '_C2O1'][:maxObs],c2DataO2[key + '_C2O2'][:maxObs],c2DataO3[key + '_C2O3'][:maxObs]))
        print('Condition ' + c2Params['train'] + ' merging ' + str(key) + ' of shapes: ' + 
              str(np.array(c2DataO1[key + "_C2O1"]).shape) + ', ' + 
              str(np.array(c2DataO2[key + "_C2O2"]).shape) + ', ' + 
              str(np.array(c2DataO3[key + "_C2O3"]).shape) + ' new shape: ' + str(np.array(c2Data[key]).shape))
        c3Data[key] = np.vstack((c3DataO1[key + '_C3O1'][:maxObs],c3DataO2[key + '_C3O2'][:maxObs],c3DataO3[key + '_C3O3'][:maxObs]))
        print('Condition ' + c3Params['train'] + ' merging ' + str(key) + ' of shapes: ' + 
              str(np.array(c3DataO1[key + "_C3O1"]).shape) + ', ' + 
              str(np.array(c3DataO2[key + "_C3O2"]).shape) + ', ' + 
              str(np.array(c3DataO3[key + "_C3O3"]).shape) + ' new shape: ' + str(np.array(c3Data[key]).shape))
        
    c1Data['act_combs'], c1Data['act_labels'] = actionProcessing(c1Data['actions'],numActs)
    c2Data['act_combs'], c2Data['act_labels'] = actionProcessing(c2Data['actions'],numActs)
    c3Data['act_combs'], c3Data['act_labels'] = actionProcessing(c3Data['actions'],numActs)
    
    return c1Data, c2Data, c3Data#c1Data, c2Data, c3Data

def getClusterVariance(cluster,data,num_cluster):
    between_var = np.var(data,axis=0)
    within_var = []
    for i in range(num_cluster):
        which = np.where(cluster==i)
        within_var.append(np.var(data[which],axis=0))
    b = np.mean(between_var)
    w = np.zeros(num_cluster)
    for i in range(num_cluster):
        w[i] = np.mean(within_var[i])
    return between_var, within_var, b, w
def getDistances(points):
    distances = [distance.euclidean(p1, p2) for p1, p2 in combinations(points, 2)]
    return distances
def getAcrossD(c1,c2):
    comb = [(x,y) for x in c1 for y in c2]
    distances = [distance.euclidean(p1, p2) for p1, p2 in comb]
    return distances
def getCorDistances(points):
    distances = [distance.correlation(p1, p2) for p1, p2 in combinations(points, 2)]
    return distances
def getAcrossCorD(c1,c2):
    comb = [(x,y) for x in c1 for y in c2]
    distances = [distance.correlation(p1, p2) for p1, p2 in comb]
    return distances
def where_array_equal(a,b):
    equal = np.zeros(a.shape[0])
    for i,arr in enumerate(a):
        if np.array_equal(arr,b):
            equal[i] = 1
    return equal
def normalize(x):
    shape = x.shape
    x = x.flatten()
    normalized = (x-min(x))/(max(x)-min(x))
    return normalized.reshape(shape)

def plot_movie_js2(enc_array,image_array,save=None):
    #Shows encoding and frames
    fig = plt.figure(figsize=(10,3), dpi=72)
    ax1 = fig.add_subplot(2, 2, 1)
    plt.title('Visual Encoding', fontsize=15)
    plt.axis('off')
    im = plt.imshow(enc_array[0][:8,:],vmin=-1,vmax=25)
    ax2 = fig.add_subplot(2, 2, 3)
    plt.title('Vector Encoding', fontsize=15)
    plt.axis('off')
    im2 = plt.imshow(enc_array[0][8:,:],vmin=-1,vmax=25)
    ax3 = fig.add_subplot(1, 2, 2)
    im3 = plt.imshow(image_array[0])
    plt.axis('off')
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

    def animate(i):
        im.set_array(enc_array[i][:8,:])
        im2.set_array(enc_array[i][8:,:])
        im3.set_array(image_array[i])
        return (im,)
    
    anim = animation.FuncAnimation(fig, animate, frames=len(image_array))

    display(IPython_display.display_animation(anim))
    if save!=None:
        anim.save(save, writer=writer)
        
def save_movie_js2(enc_array,image_array,save=None):
    #dpi = 72.0
    #xpixels, ypixels = image_array[0].shape[0], image_array[0].shape[1]
    fig = plt.figure(figsize=(10,3), dpi=72)
    ax1 = fig.add_subplot(2, 2, 1)
    plt.title('Visual Encoding', fontsize=15)
    plt.axis('off')
    im = plt.imshow(enc_array[0][:8,:],vmin=-1,vmax=25)
    ax2 = fig.add_subplot(2, 2, 3)
    plt.title('Vector Encoding', fontsize=15)
    plt.axis('off')
    im2 = plt.imshow(enc_array[0][8:,:],vmin=-1,vmax=25)
    ax3 = fig.add_subplot(1, 2, 2)
    im3 = plt.imshow(image_array[0])
    plt.axis('off')

    for i in range(enc_array.shape[0]):
        im.set_array(enc_array[i][:8,:])
        im2.set_array(enc_array[i][8:,:])
        im3.set_array(image_array[i])
        #plt.savefig(path+save+'/img'+str(i).zfill(4)+'.png', bbox_inches='tight')
        plt.savefig(save+'/img'+str(i)+'.png', bbox_inches='tight')
        
def plot_movie_jsInfo(enc_array,image_array,acts,vals,rews,onlyVisual=False,save=None):
    
    def getImage(act,num):
        if act==0:
            return stand
        if num==0:
            if act==1:
                return up
            if act==2:
                return down
        if num==1:
            if act==1:
                return turn_l
            if act==2:
                return turn_r
        if num==2 and act==1:
            return jump
        if num==3:
            if act==1:
                return right
            if act==2:
                return left
    
    jump = imageio.imread('./symbols/jump.png')
    left = imageio.imread('./symbols/arrow-left.png')
    right = imageio.imread('./symbols/arrow_right.png')
    down = imageio.imread('./symbols/down-arrow.png')
    up = imageio.imread('./symbols/up-arrow.png')
    turn_l = imageio.imread('./symbols/turn-left.png')
    turn_r = imageio.imread('./symbols/turn-right.png')
    stand = imageio.imread('./symbols/Stand.png')
    
    fig = plt.figure(figsize=(10,3), dpi=72)
    fig.subplots_adjust(wspace=0)
    if onlyVisual:
        ax1 = fig.add_subplot(1, 2, 1)
    else:
        ax1 = fig.add_subplot(2, 2, 1)
    plt.axis('off')
    if not isinstance(enc_array,list):
        if onlyVisual:
            im = plt.imshow(enc_array[0].reshape(8,int(enc_array[0].shape[0]/8)),vmin=-1,
                            vmax=np.mean(enc_array[enc_array>1]))
        else:
            im = plt.imshow(enc_array[0][:8,:],vmin=-1,vmax=25)
        plt.title('Visual Encoding', fontsize=15)
    else:
        icaLen = enc_array[0][0].shape[0]
        im = plt.imshow(enc_array[0][0].reshape(1,icaLen),vmin=-1,vmax=25)
        plt.title('Visual Encoding - ICs', fontsize=15)
    if onlyVisual:
        ax2 = fig.add_subplot(1, 2, 2)
    else:
        ax2 = fig.add_subplot(2, 2, 2)
    plt.axis('off')
    if not onlyVisual:
        if not isinstance(enc_array,list):
            im2 = plt.imshow(enc_array[0][8:,:],vmin=-1,vmax=25)
            plt.title('Vector Encoding', fontsize=15)
        else:
            im2 = plt.imshow(enc_array[1][0].reshape(1,icaLen),vmin=-1,vmax=25)
            plt.title('Vector Encoding - ICs', fontsize=15)
    ax4 = fig.add_subplot(1, 2, 2)
    im4 = plt.imshow(image_array[0])
    plt.axis('off')
    ax3 = fig.add_subplot(6, 2, 2)
    im3 = plt.text(0.3,0.1,"R: "+str(rews[0])+' V: '+str(vals[0]), fontsize=15,color='white',
                   bbox=dict(facecolor='blue', alpha=0.5))
    plt.axis('off')
    
    ax5 = fig.add_subplot(4, 10, 10)
    im5 = plt.imshow(getImage(acts[0][0][0],0))
    plt.axis('off')
    ax6 = fig.add_subplot(4, 10, 20)
    im6 = plt.imshow(getImage(acts[0][0][1],1))
    plt.axis('off')
    ax7 = fig.add_subplot(4, 10, 30)
    im7 = plt.imshow(getImage(acts[0][0][2],2))
    plt.axis('off')
    ax8 = fig.add_subplot(4, 10, 40)
    im8 = plt.imshow(getImage(acts[0][0][3],3))
    plt.axis('off')
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    
    def animate(i):
        if not isinstance(enc_array,list):
            if onlyVisual:
                im.set_array(enc_array[i].reshape(8,int(enc_array[0].shape[0]/8)))
            else:
                im.set_array(enc_array[i][:8,:])
                im2.set_array(enc_array[i][8:,:])
        else:
            im.set_array(enc_array[0][i].reshape(1,icaLen))
            im2.set_array(enc_array[1][i].reshape(1,icaLen))
        im3.set_text("R: "+str(rews[i])[:3]+' V: '+str(vals[i][0][0])[:4])
        im4.set_array(image_array[i])
        im5.set_array(getImage(acts[i][0][0],0))
        im6.set_array(getImage(acts[i][0][1],1))
        im7.set_array(getImage(acts[i][0][2],2))
        im8.set_array(getImage(acts[i][0][3],3))
        return (im,)

    anim = animation.FuncAnimation(fig, animate, frames=len(image_array))
    display(IPython_display.display_animation(anim))
    if save!=None:
        anim.save(save, writer=writer)

def save_movie_jsInfo(enc_array,image_array,acts,vals,rews,save=None):
    
    def getImage(act,num):
        if act==0:
            return stand
        if num==0:
            if act==1:
                return up
            if act==2:
                return down
        if num==1:
            if act==1:
                return turn_l
            if act==2:
                return turn_r
        if num==2 and act==1:
            return jump
        if num==3:
            if act==1:
                return right
            if act==2:
                return left
    
    jump = imageio.imread('./symbols/jump.png')
    left = imageio.imread('./symbols/arrow-left.png')
    right = imageio.imread('./symbols/arrow_right.png')
    down = imageio.imread('./symbols/down-arrow.png')
    up = imageio.imread('./symbols/up-arrow.png')
    turn_l = imageio.imread('./symbols/turn-left.png')
    turn_r = imageio.imread('./symbols/turn-right.png')
    stand = imageio.imread('./symbols/Stand.png')
    
    fig = plt.figure(figsize=(10,3), dpi=72)
    ax1 = fig.add_subplot(2, 2, 1)
    plt.axis('off')
    if not isinstance(enc_array,list):
        im = plt.imshow(enc_array[0][:8,:],vmin=-1,vmax=25)
        plt.title('Visual Encoding', fontsize=15)
    else:
        icaLen = enc_array[0][0].shape[0]
        im = plt.imshow(enc_array[0][0].reshape(1,icaLen),vmin=-1,vmax=25)
        plt.title('Visual Encoding - ICs', fontsize=15)
    ax2 = fig.add_subplot(2, 2, 3)
    plt.axis('off')
    if not isinstance(enc_array,list):
        im2 = plt.imshow(enc_array[0][8:,:],vmin=-1,vmax=25)
        plt.title('Vector Encoding', fontsize=15)
    else:
        im2 = plt.imshow(enc_array[1][0].reshape(1,icaLen),vmin=-1,vmax=25)
        plt.title('Vector Encoding - ICs', fontsize=15)
    ax4 = fig.add_subplot(1, 2, 2)
    im4 = plt.imshow(image_array[0])
    plt.axis('off')
    ax3 = fig.add_subplot(6, 2, 2)
    im3 = plt.text(0.2,0.1,"R: "+str(rews[0])+' V: '+str(vals[0]), fontsize=15,color='white',
                   bbox=dict(facecolor='blue', alpha=0.5))
    plt.axis('off')
    
    ax5 = fig.add_subplot(4, 10, 10)
    im5 = plt.imshow(getImage(acts[0][0][0],0))
    plt.axis('off')
    ax6 = fig.add_subplot(4, 10, 20)
    im6 = plt.imshow(getImage(acts[0][0][1],1))
    plt.axis('off')
    ax7 = fig.add_subplot(4, 10, 30)
    im7 = plt.imshow(getImage(acts[0][0][2],2))
    plt.axis('off')
    ax8 = fig.add_subplot(4, 10, 40)
    im8 = plt.imshow(getImage(acts[0][0][3],3))
    plt.axis('off')
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    
    if isinstance(enc_array,list):
        a_len = rews.shape[0]
    else:
        a_len = enc_array.shape[0]
    
    for i in range(a_len):
        if not isinstance(enc_array,list):
            im.set_array(enc_array[i][:8,:])
            im2.set_array(enc_array[i][8:,:])
        else:
            im.set_array(enc_array[0][i].reshape(1,icaLen))
            im2.set_array(enc_array[1][i].reshape(1,icaLen))
        im3.set_text("R: "+str(rews[i])[:3]+' V: '+str(vals[i][0][0])[:4])
        im4.set_array(image_array[i])
        im5.set_array(getImage(acts[i][0][0],0))
        im6.set_array(getImage(acts[i][0][1],1))
        im7.set_array(getImage(acts[i][0][2],2))
        im8.set_array(getImage(acts[i][0][3],3))
        plt.savefig(save+'/img'+str(i).zfill(4)+'.png', bbox_inches='tight')

def plot_movie_js3(enc_array,image_array,cluster, save=None):
    #Plot Encodings and frames + information about which cluster the frame is in
    fig = plt.figure(figsize=(10,3), dpi=72)
    ax1 = fig.add_subplot(2, 2, 1)
    plt.axis('off')
    im = plt.imshow(enc_array[0][:8,:],vmin=-1,vmax=25)
    plt.title('Visual Encoding', fontsize=15)
    ax2 = fig.add_subplot(2, 2, 3)
    plt.axis('off')
    im2 = plt.imshow(enc_array[0][8:,:],vmin=-1,vmax=25)
    plt.title('Vector Encoding', fontsize=15)
    ax4 = fig.add_subplot(1, 2, 2)
    plt.axis('off')
    im4 = plt.imshow(image_array[0])
    ax3 = fig.add_subplot(6, 2, 2)
    im3 = plt.text(0.3,0.1,'Cluster ' + str(cluster[0]), fontsize=20,color='white',bbox=dict(facecolor='blue', alpha=0.5))
    plt.axis('off')
    
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

    def animate(i):
        im.set_array(enc_array[i][:8,:])
        im2.set_array(enc_array[i][8:,:])
        im3.set_text('Cluster ' + str(cluster[i]))
        im4.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(fig, animate, frames=len(image_array))
    display(IPython_display.display_animation(anim))
    if save!=None:
        anim.save(save, writer=writer)

def plot_actions(act):
    fig = plt.figure(figsize=(7,7))
    fig.suptitle('Distribution of Actions', fontsize=20)
    ax1 = fig.add_subplot(2, 2, 1)
    plt.hist(act[:,0,0],bins=np.linspace(-0.4,2.6,4),color='chocolate',width=0.8)
    plt.title('Move Forward/Back', fontsize=15)
    plt.xticks(np.arange(3),['Stand','Forward','Back'], fontsize=13)
    ax2 = fig.add_subplot(2, 2, 2)
    plt.hist(act[:,0,1],bins=np.linspace(-0.4,2.6,4),color='chocolate',width=0.8)
    plt.title('Camera', fontsize=15)
    plt.xticks(np.arange(3),['Straight','Left','Right'], fontsize=13)
    ax3 = fig.add_subplot(2, 2, 3)
    plt.hist(act[:,0,2],bins=np.linspace(-0.4,1.6,3),color='chocolate',width=0.8)
    plt.title('Jump', fontsize=15)
    plt.xticks(np.arange(2),['Stand','Jump'], fontsize=13)
    ax4 = fig.add_subplot(2, 2, 4)
    plt.hist(act[:,0,3],bins=np.linspace(-0.4,2.6,4),color='chocolate',width=0.8)
    plt.title('Move Left/Right', fontsize=15)
    plt.xticks(np.arange(3),['Stand','Right','Left'], fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def plot_actions2(act,labels):
    colors = [np.array([157, 195, 230])/255,np.array([169, 209, 142])/255,np.array([255, 217, 102])/255]
    fig = plt.figure(figsize=(7,7))
    fig.suptitle('Distribution of Actions', fontsize=20)
    ax1 = fig.add_subplot(2, 2, 1)
    plt.hist(act[:,:,0],bins=np.linspace(-0.4,2.6,4),color=colors,width=0.26)
    plt.title('Move Forward/Back', fontsize=15)
    plt.xticks(np.arange(3),['Stand','Forward','Back'], fontsize=13)
    ax2 = fig.add_subplot(2, 2, 2)
    plt.hist(act[:,:,1],bins=np.linspace(-0.4,2.6,4),color=colors,width=0.26)
    plt.title('Camera', fontsize=15)
    plt.xticks(np.arange(3),['Straight','Left','Right'], fontsize=13)
    plt.legend(labels, fontsize=12, loc=2)
    ax3 = fig.add_subplot(2, 2, 3)
    plt.hist(act[:,:,2],bins=np.linspace(-0.4,1.6,3),color=colors,width=0.26)
    plt.title('Jump', fontsize=15)
    plt.xticks(np.arange(2),['Stand','Jump'], fontsize=13)
    ax4 = fig.add_subplot(2, 2, 4)
    plt.hist(act[:,:,3],bins=np.linspace(-0.4,2.6,4),color=colors,width=0.26)
    plt.title('Move Left/Right', fontsize=15)
    plt.xticks(np.arange(3),['Stand','Right','Left'], fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def getACombLabel(combination,lineBreak=False):
    label = ""
    actionNames = {0:['Stand','Forward','Backward'], 1:['Camera Straight','Camera Left','Camera Right'],
                   2:['Stand','Jump'],3:['Stand','Move Right','Move Left']}
    for b in range(combination.shape[0]):
        if combination[b]>0:
            if b == 0 or len(label) == 0:
                label = label + actionNames[b][combination[b]]
            else:
                if lineBreak:
                    label = label + '\n' + actionNames[b][combination[b]]
                else:
                    label = label + ', ' + actionNames[b][combination[b]]
    return label

def getAllACombs(a_vec,sort,numCombs=5):
    all_a_comb = np.ones(a_vec.shape[-1])
    all_a_comb = all_a_comb*numCombs
    for i in range(a_vec.shape[-1]):
        for comb in range(numCombs):
            if a_vec[sort[comb]][i] == 1:
                all_a_comb[i] = comb
    return all_a_comb

def actionProcessing(actions,numCombs=5):
    combs = np.unique(actions[:,0],axis=0)
    a_vec = np.zeros((combs.shape[0],actions.shape[0]))
    for i,c in enumerate(combs):
        a_vec[i] = where_array_equal(actions[:,0],c)
    count = np.sum(a_vec,axis=1)
    sort = np.argsort(count)[::-1]
    all_a_comb = getAllACombs(a_vec,sort,numCombs)
    labels = []
    for a in combs[sort[:numCombs]]:
        labels.append(getACombLabel(a))
    labels.append('Everything Else')
    return all_a_comb,labels

def plotActCombs(actions):
    combs = np.unique(actions[:,0],axis=0)
    a_vec = np.zeros((combs.shape[0],actions.shape[0]))
    for i,c in enumerate(combs):
        a_vec[i] = where_array_equal(actions[:,0],c)
        
    fig = plt.figure(figsize=(17,5))
    plt.subplots_adjust(hspace=0.2,wspace=0.1)
    ax1 = plt.subplot(1,2,1)
    
    plt.title("Distribution of Frames for all Action Combinations",fontsize=15)
    
    count = np.sum(a_vec,axis=1)
    sort = np.argsort(count)[::-1]
    colors = np.zeros(combs.shape[0],dtype=int)
    pallete = np.ones(combs.shape[0])*7
    pallete[:5] = [0,1,3,4,6]
    for i,e in enumerate(sort):
        colors[e] = int(pallete[i])

    p1 = np.array(sns.color_palette("Accent", n_colors=8))[np.array(colors)]#([0,7,7,1,3,7,4,7,7,7,7,7,7,6,7])

    b1 = sns.barplot(x=np.linspace(0,combs.shape[0]-1,combs.shape[0]),y=count,palette=p1)

    plt.xlabel("Action Combination",fontsize=15)
    plt.ylabel("# of Frames",fontsize=15)
    plt.xticks(np.linspace(0,combs.shape[0]-1,combs.shape[0]),np.linspace(0,combs.shape[0]-1,combs.shape[0],dtype=int))
    
    ax2 = plt.subplot(1,2,2)
    plt.title("Distribution of Frames for Selected Action Combinations",fontsize=15)
    
    all_a_comb = getAllACombs(a_vec,sort)

    p2 = np.array(sns.color_palette("Accent", n_colors=8))[np.array([0,1,3,4,6,7])]
    a2 = np.histogram(all_a_comb,bins=np.linspace(-0.5,5.5,7))
    b2 = sns.barplot(x=np.linspace(0,5,6),y=a2[0],palette=p2,ax=ax2)

    #plt.xlabel("Action Combination",fontsize=15)
    #plt.ylabel("# of Frames",fontsize=15)
    
    labels = []
    for c in sort[:5]:
        labels.append(getACombLabel(combs[c],lineBreak=True))
    labels.append('Everything\nElse')
    plt.xticks(np.linspace(0,5,6),labels)
    return fig

def getEdgeParams(labels):
    edgeWs = np.array(labels)
    edgeWs[edgeWs == 0]=0
    edgeWs[edgeWs == 1]=0
    edgeWs[edgeWs == 2]=0
    edgeWs[edgeWs == 3]=0
    edgeWs[edgeWs == 4]=0
    edgeWs[edgeWs == 5]=0
    edgeWs[edgeWs == 6]=1
    edgeWs[edgeWs == 7]=0
    edgeWs[edgeWs == 8]=0
    edgeWs[edgeWs == 9]=0
    
    edgeCs = np.array(labels)
    edgeCs = edgeCs.astype(str)
    edgeCs[edgeCs=='0'] = 'b'
    edgeCs[edgeCs=='1'] = 'b'
    edgeCs[edgeCs=='2'] = 'b'
    edgeCs[edgeCs=='3'] = 'green'
    edgeCs[edgeCs=='4'] = 'firebrick'
    edgeCs[edgeCs=='5'] = 'purple'
    edgeCs[edgeCs=='9'] = 'b'
    edgeCs[edgeCs=='6'] = 'red'
    edgeCs[edgeCs=='7'] = 'yellow'
    edgeCs[edgeCs=='8'] = 'darkblue'

    newOrder = np.append(np.where(edgeWs==0),np.where(edgeWs!=0))#to plot door frames on top (better visibility)
    edgeWs[newOrder]
    return edgeWs, edgeCs, newOrder


def correlate(enc,val,num=0,normalize=False):
    corrs = []
    v = val#[:,0,num]
    if normalize:
        v = (v - np.mean(v)) /  np.std(v)
    for i in range(enc.shape[-1]):
        e = enc[:,i]
        if np.std(e) == 0:
            corrs.append(0.0)
        else:
            if normalize:
                e = (e - np.mean(e)) / (np.std(e) * len(e))
            #corr = np.correlate(e,v)
            corr = np.corrcoef(e,v)[0,1]
            corrs.append(float(corr))
    return np.array(corrs)

def plot_movie_curInfo(enc_array,image_array,pred_s,enc_cur,acts,vals,rews,range_vis,range_vec,save=None):
    
    fig = plt.figure(figsize=(10,3), dpi=72)
    ax1 = fig.add_subplot(3, 3, 1)
    plt.axis('off')
    if not isinstance(enc_array,list):
        im = plt.imshow(enc_array[0][:4,:],vmin=range_vis[0], vmax=range_vis[1])
        plt.title('Visual Encoding', fontsize=15)
    else:
        icaLen = enc_array[0][0].shape[0]
        im = plt.imshow(enc_array[0][0].reshape(1,icaLen))
        plt.title('Visual Encoding - ICs', fontsize=15)
    ax2 = fig.add_subplot(3, 3, 2)
    plt.axis('off')
    if not isinstance(enc_array,list):
        im2 = plt.imshow(enc_array[0][4:,:],vmin=range_vec[0], vmax=range_vec[1])
        plt.title('Vector Encoding', fontsize=15)
    else:
        im2 = plt.imshow(enc_array[1][0].reshape(1,icaLen))
        plt.title('Vector Encoding - ICs', fontsize=15)
       
        
    ax5 = fig.add_subplot(3,3,4)
    plt.axis('off')
    im5 = plt.imshow(pred_s[0][:4,:],vmin=range_vis[0], vmax=range_vis[1])
    plt.title('Predicted')
    
    ax6 = fig.add_subplot(3,3,5)
    plt.axis('off')
    im6 = plt.imshow(pred_s[0][4:,:],vmin=range_vec[0], vmax=range_vec[1])
    plt.title('Predicted')
    
    ax7 = fig.add_subplot(3,3,7)
    plt.axis('off')
    im7 = plt.imshow(enc_cur[0][:4,:],vmin=range_vis[0], vmax=range_vis[1])
    plt.title('Actual')
    
    ax8 = fig.add_subplot(3,3,8)
    plt.axis('off')
    im8 = plt.imshow(enc_cur[0][4:,:],vmin=range_vec[0], vmax=range_vec[1])
    plt.title('Actual')
    
    ax4 = fig.add_subplot(1, 3, 3)
    im4 = plt.imshow(image_array[0])
    plt.axis('off')
    ax3 = fig.add_subplot(6, 3, 3)
    im3 = plt.text(0.2,0.1,"R: "+str(rews[0])+' V: '+str(vals[0]), fontsize=15,color='white',
                   bbox=dict(facecolor='blue', alpha=0.5))
    plt.axis('off')
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    
    def animate(i):
        if not isinstance(enc_array,list):
            im.set_array(enc_array[i][:4,:])
            im2.set_array(enc_array[i][4:,:])
        else:
            im.set_array(enc_array[0][i].reshape(1,icaLen))
            im2.set_array(enc_array[1][i].reshape(1,icaLen))
        im3.set_text("R: "+str(rews[i])[:3]+' V: '+str(vals[i][0][0])[:4])
        im4.set_array(image_array[i])
        
        im5.set_array(pred_s[i][:4,:])
        im6.set_array(pred_s[i][4:,:])
        im7.set_array(enc_cur[i][:4,:])
        im8.set_array(enc_cur[i][4:,:])
        return (im,)

    anim = animation.FuncAnimation(fig, animate, frames=len(image_array))
    display(IPython_display.display_animation(anim))
    if save!=None:
        anim.save(save, writer=writer)
        
def plot_movie_semantic(semantic,image_array,acts,vals,rews,save=None):
    
    def getImage(act,num):
        if act==0:
            return stand
        if num==0:
            if act==1:
                return up
            if act==2:
                return down
        if num==1:
            if act==1:
                return turn_l
            if act==2:
                return turn_r
        if num==2 and act==1:
            return jump
        if num==3:
            if act==1:
                return right
            if act==2:
                return left
    
    jump = imageio.imread('./symbols/jump.png')
    left = imageio.imread('./symbols/arrow-left.png')
    right = imageio.imread('./symbols/arrow_right.png')
    down = imageio.imread('./symbols/down-arrow.png')
    up = imageio.imread('./symbols/up-arrow.png')
    turn_l = imageio.imread('./symbols/turn-left.png')
    turn_r = imageio.imread('./symbols/turn-right.png')
    stand = imageio.imread('./symbols/Stand.png')
    
    fig = plt.figure(figsize=(10,3), dpi=72)
    ax1 = fig.add_subplot(1, 2, 1)
    plt.axis('off')
    lbls = np.rot90(np.array([int(n) for n in semantic[0][1:-4].split(",")]).reshape((128,128)))
    im1 = plt.imshow(lbls,vmin=-1,vmax=10,cmap='tab20')
    values = np.linspace(-1,10,12)
    colors = [ im1.cmap(im1.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label=inv_map[values[i]] ) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    ax4 = fig.add_subplot(1, 2, 2)
    im4 = plt.imshow(image_array[0])
    plt.axis('off')
    ax3 = fig.add_subplot(6, 2, 2)
    im3 = plt.text(0.2,0.1,"R: "+str(rews[0])+' V: '+str(vals[0]), fontsize=15,color='white',
                   bbox=dict(facecolor='blue', alpha=0.5))
    plt.axis('off')
    
    ax5 = fig.add_subplot(4, 10, 10)
    im5 = plt.imshow(getImage(acts[0][0][0],0))
    plt.axis('off')
    ax6 = fig.add_subplot(4, 10, 20)
    im6 = plt.imshow(getImage(acts[0][0][1],1))
    plt.axis('off')
    ax7 = fig.add_subplot(4, 10, 30)
    im7 = plt.imshow(getImage(acts[0][0][2],2))
    plt.axis('off')
    ax8 = fig.add_subplot(4, 10, 40)
    im8 = plt.imshow(getImage(acts[0][0][3],3))
    plt.axis('off')
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    

    
    def animate(i):
        try:
            lbls = np.rot90(np.array([int(n) for n in semantic[i][1:-4].split(",")]).reshape((128,128)))
            im1.set_array(lbls)
        except:
            broken = np.array([int(n) for n in data[i][1:-4].split(",")])
            lbls = np.rot90(np.append(broken,np.zeros((128*128)-broken.shape[0])).reshape((128,128)))
            im1.set_array(lbls)
            print(str(i)+" - "+str(broken.shape))
        
        im3.set_text("R: "+str(rews[i])[:3]+' V: '+str(vals[i][0][0])[:4])
        im4.set_array(image_array[i])
        im5.set_array(getImage(acts[i][0][0],0))
        im6.set_array(getImage(acts[i][0][1],1))
        im7.set_array(getImage(acts[i][0][2],2))
        im8.set_array(getImage(acts[i][0][3],3))
        return (im1,)

    anim = animation.FuncAnimation(fig, animate, frames=len(image_array))
    display(IPython_display.display_animation(anim))
    if save!=None:
        anim.save(save, writer=writer)
        
def plot_movie_semantic2(semantic,image_array,acts,vals,rews,save=None):
    label_dict = {"Unknown": 0,
              "Agent": 1,
          "Level Door": 2,
          "Regular Door": 3 ,
          "Key Door": 4 ,
          "Entry Door": 5 ,
          "Puzzle Door": 6 ,
          "Key": 7 ,
          "Time Orb": 8 ,
          "Wall":9,
          "Floor": 10}
    inv_map = {v: k for k, v in label_dict.items()}
    def getImage(act,num):
        if act==0:
            return stand
        if num==0:
            if act==1:
                return up
            if act==2:
                return down
        if num==1:
            if act==1:
                return turn_l
            if act==2:
                return turn_r
        if num==2 and act==1:
            return jump
        if num==3:
            if act==1:
                return right
            if act==2:
                return left
    
    jump = imageio.imread('./symbols/jump.png')
    left = imageio.imread('./symbols/arrow-left.png')
    right = imageio.imread('./symbols/arrow_right.png')
    down = imageio.imread('./symbols/down-arrow.png')
    up = imageio.imread('./symbols/up-arrow.png')
    turn_l = imageio.imread('./symbols/turn-left.png')
    turn_r = imageio.imread('./symbols/turn-right.png')
    stand = imageio.imread('./symbols/Stand.png')
    
    fig = plt.figure(figsize=(10,3), dpi=72)
    ax1 = fig.add_subplot(1, 2, 1)
    plt.axis('off')
    im1 = plt.imshow(rgb2L(semantic[0]),vmin=0,vmax=11,cmap='tab20')
    values = np.linspace(0,10,11)
    colors = [ im1.cmap(im1.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label=inv_map[values[i]] ) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    ax4 = fig.add_subplot(1, 2, 2)
    im4 = plt.imshow(image_array[0])
    plt.axis('off')
    ax3 = fig.add_subplot(6, 2, 2)
    im3 = plt.text(0.2,0.1,"0 R: "+str(rews[0])+' V: '+str(vals[0]), fontsize=15,color='white',
                   bbox=dict(facecolor='blue', alpha=0.5))
    plt.axis('off')
    
    ax5 = fig.add_subplot(4, 10, 10)
    im5 = plt.imshow(getImage(acts[0][0][0],0))
    plt.axis('off')
    ax6 = fig.add_subplot(4, 10, 20)
    im6 = plt.imshow(getImage(acts[0][0][1],1))
    plt.axis('off')
    ax7 = fig.add_subplot(4, 10, 30)
    im7 = plt.imshow(getImage(acts[0][0][2],2))
    plt.axis('off')
    ax8 = fig.add_subplot(4, 10, 40)
    im8 = plt.imshow(getImage(acts[0][0][3],3))
    plt.axis('off')
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    

    
    def animate(i):
        try:
            im1.set_array(rgb2L(semantic[i]))
        except:
            print(str(i)+" - "+str(broken.shape))
        
        im3.set_text(str(i)+" R: "+str(rews[i])[:3]+' V: '+str(vals[i][0][0])[:4])
        im4.set_array(image_array[i])
        im5.set_array(getImage(acts[i][0][0],0))
        im6.set_array(getImage(acts[i][0][1],1))
        im7.set_array(getImage(acts[i][0][2],2))
        im8.set_array(getImage(acts[i][0][3],3))
        return (im1,)

    anim = animation.FuncAnimation(fig, animate, frames=len(image_array))
    display(IPython_display.display_animation(anim))
    if save!=None:
        anim.save(save, writer=writer)

def rgb2L(img):
    l_img = np.zeros((img.shape[0]*img.shape[1]))
    for i,p in enumerate(img.reshape(img.shape[0]*img.shape[1],3)):
        if (p[0] in range(5,15) and p[1] in range(0,2) and p[2] in range(65,75)):
            l_img[i] = 1#Agent
        elif(p[0] in range(60,70) and p[1] in range(62,67) and p[2] in range(22,30)):
            l_img[i] = 2#Level Door
        elif(p[0] in range(30,50) and p[1] in range(65,105) and p[2] in range(30,50)):
            l_img[i] = 3#Green Door
        elif(p[0] in range(60,70) and p[1] in range(30,35) and p[2] in range(30,35)):
            l_img[i] = 4#Key Door
        elif(p[0] in range(27,32) and p[1] in range(30,35) and p[2] in range(25,45)):
            l_img[i] = 5#Entry Door
        elif(p[0] in range(55,80) and p[1] in range(35,50) and p[2] in range(75,110)):
            l_img[i] = 6#Puzzle Door
        elif(p[0] in range(50,70) and p[1] in range(60,80) and p[2] in range(0,20)):
            l_img[i] = 7#Key
        elif(p[0] in range(5,15) and p[1] in range(60,74) and p[2] in range(65,85)):
            l_img[i] = 8#Orb
        elif(p[0] in range(0,2) and p[1] in range(0,2) and p[2] in range(0,2)):
            l_img[i] = 9#Wall
        elif(p[0] in range(45,60) and p[1] in range(38,50) and p[2] in range(22,26)):
            l_img[i] = 10#Floor
        else:
            l_img[i] = 0#Other
    return l_img.reshape((img.shape[0],img.shape[1]))