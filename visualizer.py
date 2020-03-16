import os
import numpy as np
import pandas as pd
from math import sqrt, ceil
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from util import get_centroids, dm_centroid, get_sorted_idx, get_sorted_intersect, load_model, NOISETYPES, DATASETS

def plot(vals, title, xlabel='# iterations', save=True, save_folder='./', show=False):
    filename = save_folder + title+'.png'
    plt.figure()
    for name in vals:
        plt.plot(vals[name], label=name)
    plt.legend(loc='best')
    plt.xlabel(xlabel)
    plt.title(title)
    if save:
        plt.savefig(filename)
    if show is not False:
        plt.show()
    return filename

def plot_acc_loss(csv_file, save=True, save_folder='./', show=False):
    df = pd.read_csv(csv_file)
    accs = {'acc': df['acc'],
            'val_acc': df['val_acc']}
    losses = {'loss': df['loss'],
              'val_loss': df['val_loss']}
    acc_file=plot(accs, 'accuracy', save=save, save_folder=save_folder, show=show)
    loss_file=plot(losses, 'loss', save=save, save_folder=save_folder, show=show)
    return acc_file, loss_file

def plot_cm(model, x_test, y_test_int, class_names, path=None, **kwargs):
    pred = model.predict(x_test)
    pred_int = np.argmax(pred, axis=1)
    cm = confusion_matrix(y_test_int,pred_int)
    plot_matrix(cm, class_names, **kwargs)
    if path is not None:
        plt.savefig(path)
        plt.close()
    return cm

def plot_dm(model, x_test, y_test_int, class_names, path=None, **kwargs):
    logits = model.predict(x_test)
    centroids = get_centroids(logits, y_test_int ,len(class_names))
    dm_centroids = dm_centroid(centroids)
    plot_matrix(dm_centroids, class_names, **kwargs)
    if path is not None:
        plt.savefig(path)
        plt.close()
    return dm_centroids

def plot_matrix(cm, classes=None,
                normalize=True,
                title=None,
                cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Set classes
    if classes is None:
        classes = np.arange(cm.shape[0])

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    fig.set_size_inches(cm.shape[0]/2, cm.shape[1]/2)

    # correction for matplotlib bug
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom+0.5, top-0.5)

    return ax

def plot_confused_samples(probs, dataset, class_id_true=None, class_id_false=None, path=None, num_plots=100):
    y_clean, y_noisy = dataset.y_train_int(), dataset.y_noisy_int()
    class_names = dataset.class_names
    idx_mislabeled = dataset.idx_noisy
    num_plot = ceil(sqrt(num_plots))
    num_plots = num_plot * num_plot

    idx_mislabeled = np.where(y_noisy != y_clean)[0]
    if class_id_true is None or class_id_false is None:
        prob_false = np.choose(y_noisy[idx_mislabeled], probs[idx_mislabeled].T)
        idx_sorted = get_sorted_idx(prob_false, np.arange(len(y_clean[idx_mislabeled])))
        idx_mislabeled_sorted = idx_mislabeled[idx_sorted]
    else:
        idx_sorted = get_sorted_idx(probs[:,class_id_false], y_clean, class_id_true)
        idx_mislabeled_sorted = get_sorted_intersect(idx_sorted, idx_mislabeled, probs[:,class_id_false])

    fig, axs = plt.subplots(num_plot, num_plot)
    fig2, axs2 = plt.subplots(num_plot, num_plot)

    count = 0
    for j in range(num_plot):
        for k in range(num_plot):
            # confident samples for class i
            idx = idx_mislabeled_sorted[-count-1]
            axs[j, k].imshow(dataset.x_train_img()[idx], cmap='gray')
            title = '{}({:.2f}) - {}({:.2f})'.format(class_names[y_clean[idx]], probs[idx,y_clean[idx]], 
                                             class_names[y_noisy[idx]], probs[idx,y_noisy[idx]])
            axs[j, k].set_title(title)
            # unconfident samples for class i
            idx = idx_mislabeled_sorted[num_plots-count]
            axs2[j, k].imshow(dataset.x_train_img()[idx], cmap='gray')
            title = '{}({:.2f}) - {}({:.2f})'.format(class_names[y_clean[idx]], probs[idx,y_clean[idx]], 
                                             class_names[y_noisy[idx]], probs[idx,y_noisy[idx]])
            axs2[j, k].set_title(title)

            count += 1

    fig.set_size_inches(3*num_plot, 3*num_plot)
    plt.figure(fig.number)
    plt.savefig(path+'mislabeled_conf.png')
    plt.close()
    fig2.set_size_inches(3*num_plot, 3*num_plot)
    plt.figure(fig2.number)
    plt.savefig(path+'mislabeled_unconf.png')
    plt.close()

def plot_confused_samples2(probs, dataset, path, num_plots=100):
    y_clean, y_noisy = dataset.y_train_int(), dataset.y_noisy_int()
    class_names = dataset.class_names
    num_plot = ceil(sqrt(num_plots))
    num_plots = num_plot * num_plot

    for i in range(dataset.num_classes):
        idx_sorted = get_sorted_idx(probs[:,i], y_clean, i)
        fig, axs = plt.subplots(num_plot, num_plot)
        fig2, axs2 = plt.subplots(num_plot, num_plot)
        count = 0
        for j in range(num_plot):
            for k in range(num_plot):
                # confident samples for class i
                idx = idx_sorted[-count-1]
                axs[j, k].imshow(dataset.x_train_img()[idx], cmap='gray')
                title = '{}({:.2f}) - {}({:.2f})'.format(class_names[y_clean[idx]], probs[idx,y_clean[idx]], 
                                                 class_names[y_noisy[idx]], probs[idx,y_noisy[idx]])
                axs[j, k].set_title(title)
                # unconfident samples for class i
                idx = idx_sorted[num_plots-count]
                axs2[j, k].imshow(dataset.x_train_img()[idx], cmap='gray')
                title = '{}({:.2f}) - {}({:.2f})'.format(class_names[y_clean[idx]], probs[idx,y_clean[idx]], 
                                                 class_names[y_noisy[idx]], probs[idx,y_noisy[idx]])
                axs2[j, k].set_title(title)

                count += 1

        fig.set_size_inches(3*num_plot, 3*num_plot)
        plt.figure(fig.number)
        plt.savefig(path+'{}_conf.png'.format(class_names[i]))
        plt.close()
        fig2.set_size_inches(3*num_plot, 3*num_plot)
        plt.figure(fig2.number)
        plt.savefig(path+'{}_unconf.png'.format(class_names[i]))
        plt.close()

def plot_overall(base_folder, dirs=None):
    acc_train_path = base_folder+'accs_train.png'
    acc_val_path = base_folder+'accs_val.png'
    acc_test_path = base_folder+'accs_test.png'
    loss_train_path = base_folder+'losses_train.png'
    loss_val_path = base_folder+'losses_val.png'
    loss_test_path = base_folder+'losses_test.png'

    fig_acc_train = plt.figure()
    fig_acc_val = plt.figure()
    fig_acc_test = plt.figure()
    fig_loss_train = plt.figure()
    fig_loss_val = plt.figure()
    fig_loss_test = plt.figure()

    cols = ['acc', 'val_acc', 'test_acc', 'loss', 'val_loss', 'test_loss']
    titles = ['Accuracy train','Accuracy validation', 'Accuracy test', 'Loss train', 'Loss validation', 'Loss Test']
    figs = [fig_acc_train, fig_acc_val, fig_acc_test, fig_loss_train, fig_loss_val, fig_loss_test]
    paths = [acc_train_path, acc_val_path, acc_test_path, loss_train_path, loss_val_path, loss_test_path]
    
    if dirs is None:
        dirs = os.listdir(base_folder)

    for folder in dirs:
        log_path = base_folder+folder+'/log.csv'
        if os.path.isfile(log_path):
            df = pd.read_csv(log_path)
            for col, fig in zip(cols, figs):
                plt.figure(fig.number)
                plt.plot(df[col], label=folder)

    for fig, title, path in zip(figs, titles, paths):
        plt.figure(fig.number)   
        plt.legend(loc='best')
        plt.xlabel('# iterations')
        plt.title(title)
        plt.savefig(path)
        plt.close()

def logs_through_nrs(base_folder, model='ce'):
    cols = ['acc', 'val_acc', 'test_acc', 'loss', 'val_loss', 'test_loss']

    dfs = []
    for col in cols:
        dfs.append(pd.DataFrame(columns=['nr']))

    dirs = os.listdir(base_folder)

    # take only dropout
    dirs_do, dirs_no_do = [], []
    for folder in dirs:
        if folder != 'logs.csv':
            if folder.endswith('_do'):
                dirs_do.append(folder)
            else:
                dirs_no_do.append(folder)

    for folder in dirs_do:
        nr_path = base_folder+folder+'/'+model+'/'
        nr = int(folder.replace('nr_','').replace('_do',''))
        for model_path in os.listdir(nr_path):
            log_path = nr_path+model_path+'/log.csv'
            if os.path.isfile(log_path):
                df_tmp = pd.read_csv(log_path)
                for col, df in zip(cols, dfs):
                    df.at[nr, model_path] = df_tmp.iloc[-1][col]#df.set_value(nr, model_path, df_tmp.iloc[-1][col])
                    df.at[nr, 'nr'] = nr#df.set_value(nr, 'nr', nr)

    for df,col in zip(dfs,cols):
        df = df.sort_values(by=['nr'])
        df.to_csv(base_folder+'logs_{}_{}.csv'.format(model,col), index=False)

    return dfs

def plot_through_nrs(base_folder, model='ce'):
    acc_train_path = base_folder+'accs_train.png'
    acc_val_path = base_folder+'accs_val.png'
    acc_test_path = base_folder+'accs_test.png'
    loss_train_path = base_folder+'losses_train.png'
    loss_val_path = base_folder+'losses_val.png'
    loss_test_path = base_folder+'losses_test.png'

    fig_acc_train = plt.figure()
    fig_acc_val = plt.figure()
    fig_acc_test = plt.figure()
    fig_loss_train = plt.figure()
    fig_loss_val = plt.figure()
    fig_loss_test = plt.figure()

    titles = ['Accuracy train','Accuracy validation', 'Accuracy test', 'Loss train', 'Loss validation', 'Loss Test']
    figs = [fig_acc_train, fig_acc_val, fig_acc_test, fig_loss_train, fig_loss_val, fig_loss_test]
    paths = [acc_train_path, acc_val_path, acc_test_path, loss_train_path, loss_val_path, loss_test_path]

    dfs = logs_through_nrs(base_folder, model)

    for fig, df in zip(figs, dfs):
        for model_name in df.columns:
            if model_name != 'nr' and model_name != 'none':
                df = df.sort_values(by=['nr'])
                plt.figure(fig.number)
                plt.plot(df[model_name], label=model_name)

    for fig, title, path in zip(figs, titles, paths):
        plt.figure(fig.number)   
        plt.legend(loc='best')
        plt.xlabel('Noise Ratio (%)')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.savefig(path)
        plt.close()

def logs_through_percentages(base_folder, model='ce', nr=35):
    cols = ['acc', 'val_acc', 'test_acc', 'loss', 'val_loss', 'test_loss']
    dirs = os.listdir(base_folder)

    # get log folders
    log_dirs = []
    percentages = []
    for folder in dirs:
        if folder.startswith('logs_') and not folder.endswith('.csv'):
            log_dirs.append(os.path.join(base_folder, folder)+'/')
            percentages.append(folder.replace('logs_',''))

    # create csvs for each log
    for log_dir in log_dirs:
        logs_through_nrs(log_dir,model)

    dfs_orgs = []
    for log_dir in log_dirs:
        df_org = pd.read_csv(log_dir+'logs_{}_test_acc.csv'.format(model))
        dfs_orgs.append(df_org)

    columns = list(df_org.columns)
    columns[0] = 'percentage'
    raw_columns = columns.copy()
    raw_columns.remove('percentage')

    dfs = []
    for col in cols:
        df = pd.DataFrame(columns=columns)
        for i,(df_org,percentage) in enumerate(zip(dfs_orgs,percentages)):
            df.set_value(i, 'percentage', percentage)
            for nt in raw_columns:
                df.set_value(i, nt, df_org.loc[df_org['nr'] == nr, nt].values[0])
        df = df.sort_values(by=['percentage'])
        dfs.append(df) 
        df.to_csv(base_folder+'logs_{}_{}.csv'.format(model,col), index=False)   

    return dfs

def get_svcca(acts1, acts2):
    import cca_core

    # flatten convolutional layers
    if len(acts1.shape) == 4:
        num_datapoints, h, w, channels = acts1.shape
        acts1 = acts1.reshape((num_datapoints*h*w, channels))
        num_datapoints, h, w, channels = acts2.shape
        acts2 = acts2.reshape((num_datapoints*h*w, channels))

    # Mean subtract activations
    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    n = min(20, len(acts1))
    svacts1 = np.dot(s1[:n]*np.eye(n), V1[:n])
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    svacts2 = np.dot(s2[:n]*np.eye(n), V2[:n])
    # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)

    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    return np.mean(svcca_results["cca_coef1"])

def plot_cosine_similarities(base_folder, dataset):
    from tensorflow.keras.layers import dot, Dropout, Dense, Activation
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    from dataset import get_data
    from models import compile_model
    from scipy.spatial.distance import cosine
    from sklearn.preprocessing import normalize

    K.clear_session()
    dataset = get_data(dataset)
    test = dataset.x_test
    dirs = os.listdir(base_folder)

    modelDict = {}
    for folder in dirs:
        if os.path.isdir(base_folder+folder):
            model = load_model(base_folder+folder+'/model/model')
            compile_model(model)
            outputs = [layer.output for layer in model.layers if isinstance(layer, Activation)][1:-1]
            outputs.append(model.input)
            modelDict[folder] = outputs

    layer_names = [layer.name.split('/',1)[0] for layer in outputs]

    for key in modelDict:
        if key != 'none':
            similarity = []
            for i in range(len(modelDict['none'])-1):
                model_layer_clean = Model(inputs=modelDict['none'][-1], outputs=modelDict['none'][i])
                output_layer_clean = model_layer_clean.predict(test)
                model_layer_noisy = Model(inputs=modelDict[key][-1], outputs=modelDict[key][i])
                output_layer_noisy = model_layer_noisy.predict(test)
                print('{}__{}__'.format(i,output_layer_noisy.shape))

                if len(output_layer_clean.shape) == 4:
                    num_datapoints, h, w, channels = output_layer_clean.shape
                    output_layer_clean = output_layer_clean.reshape((num_datapoints*h*w, channels))
                    output_layer_noisy = output_layer_noisy.reshape((num_datapoints*h*w, channels))

                sim = get_svcca(np.transpose(output_layer_clean), np.transpose(output_layer_noisy))
                similarity.append(sim)
                print('{}{} = {}'.format(key,i,sim))
                del model_layer_clean
                del model_layer_noisy
            modelDict[key] = similarity
    
    plt.figure() 
    plt.xticks(np.arange(len(layer_names)), layer_names, rotation=45)
    for key in modelDict:
        if key != 'none':
            plt.plot(modelDict[key], label=key)  
            plt.legend(loc='best')
            plt.xlabel('Layer number')
    plt.title("Similarities of Layers (%)")
    plt.show()
    plt.savefig(base_folder+'layer_sims.png')
    plt.close()

def plot_csvs(myDict, title='Title', save_path='plot.png', xlabel=None, ylabel=None):
    plt.figure() 
    for key in myDict:
        df = pd.read_csv(myDict[key]['path'])
        plt.plot(df[myDict[key]['col']], label=myDict[key]['key'])  
        plt.legend(loc='lower ')
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    #plt.show()
    plt.savefig(save_path)
    plt.close()

def merge_images(images_list, save_path='test.jpg',orientation='horizontal'):
    import sys
    from PIL import Image

    images = [Image.open(x) for x in images_list]
    widths, heights = zip(*(i.size for i in images))

    if orientation=='horizontal':
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
    else:
        max_width = max(widths)
        total_height = sum(heights)
        new_im = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0,y_offset))
            y_offset += im.size[1]

    new_im.save(save_path)

def visualize_all():
    dirs = os.listdir('./')

    # Get list of existing logs
    base_folders = []
    for folder in dirs:
        if folder in DATASETS:
            base_folders.append(folder+'/')

    # to be removed
    base_folders = ['mnist_fashion']
    for base_folder in base_folders:
        # write all log.csv files
        logs_through_percentages(base_folder)

        # get list of log folders and percentages
        log_dirs, percentages = [], []
        dirs = os.listdir(base_folder)
        for folder in dirs:
            if folder.startswith('logs_') and os.path.isdir(base_folder+folder):
                log_dirs.append(os.path.join(base_folder, folder)+'/')
                percentages.append(folder.replace('logs_',''))
        # plots through different noise rations
        for log_dir in log_dirs:
            plot_through_nrs(log_dir)
        # plot cosine similarities

if __name__ == "__main__":      
    #merge_images(['mnist_fashion_35.png','cifar100_35.png'],save_path='accuracies.png',orientation='vertical')
    #merge_images(['./cifar100/logs_1/accs_train.png','./cifar100/logs_1/accs_val.png', './cifar100/logs_1/accs_test.png'],save_path='noiseratios.png')
    
    dataset = 'mnist_fashion'
    noise_rate = 45
    cols = ['acc', 'val_acc', 'test_acc']
    titles = {'acc':'Train accuracy', 'val_acc':'Validation accuracy', 'test_acc':'Test accuracy'}
    images_list = []
    for col in cols:
        myDict = {'xy_cnn': {'path': '{}/logs_1/nr_{}_do/ce/xydistillation116/log.csv'.format(dataset,noise_rate), 'col': col, 'key':'xy_cnn'},
                  'xy_mlp': {'path': '{}_mlp/logs_1/nr_{}_do/ce/xydistillation116/log.csv'.format(dataset,noise_rate), 'col': col, 'key':'xy_mlp'},
                  'localized_cnn': {'path': '{}/logs_1/nr_{}_do/ce/xylocalized/log.csv'.format(dataset,noise_rate), 'col': col, 'key':'localized_cnn'},
                  'localized_mlp': {'path': '{}_mlp/logs_1/nr_{}_do/ce/xylocalized/log.csv'.format(dataset,noise_rate), 'col': col, 'key':'localized_mlp'},
                  'classdependent_cnn': {'path': '{}/logs_1/nr_{}_do/ce/ymodelpred/log.csv'.format(dataset,noise_rate), 'col': col, 'key':'classdependent_cnn'},
                  'classdependent_mlp': {'path': '{}_mlp/logs_1/nr_{}_do/ce/ymodelpred/log.csv'.format(dataset,noise_rate), 'col': col, 'key':'classdependent_mlp'},
                  'uniform_cnn': {'path': '{}/logs_1/nr_{}_do/ce/yuniform/log.csv'.format(dataset,noise_rate), 'col': col, 'key':'uniform_cnn'},
                  'uniform_mlp': {'path': '{}_mlp/logs_1/nr_{}_do/ce/yuniform/log.csv'.format(dataset,noise_rate), 'col': col, 'key':'uniform_mlp'}}
        save_path = '{}_{}_{}_.png'.format(dataset,noise_rate,col)
        images_list.append(save_path)
        plot_csvs(myDict,title='{} for %{} noise rate: {}'.format(titles[col],noise_rate, dataset.upper()), save_path=save_path, xlabel='# epochs', ylabel='Accuracy')
    merge_images(images_list,save_path='{}_{}.png'.format(dataset,noise_rate))    

    #plot_cosine_similarities('cifar10/logs_1/nr_45_do/ce/', 'cifar10')
    #logs_through_percentages('mnist_fashion/')
    #plot_through_nrs('mnist_fashion/logs_1/')
    #plot_overall('cifar10/logs/nr_45/ce/')